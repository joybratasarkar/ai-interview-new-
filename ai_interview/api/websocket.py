from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
import base64
import time
import logging
import torch
import numpy as np
from scipy.io import wavfile
# from ai_interview.smart_turn.inference import predict_endpoint  # Removed ML turn detection
from ai_interview.celery_app import celery_app
from ai_interview.tasks.video import process_video
from ai_interview.services.interview_flow_service import interview_flow_service
from ai_interview.services.socket_manager import manager
import redis.asyncio as aioredis
from ai_interview.config import REDIS_URL

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Config --- SILENCE-BASED TURN DETECTION
RATE = 16000
CHUNK = 512  # Standard chunk size for stable audio processing
STOP_MS = 1000  # Silence duration for turn detection (2 seconds)
PRE_SPEECH_MS = 150  # Pre-speech buffer
MAX_DURATION_SECONDS = 16
VAD_THRESHOLD = 0.5  # VAD threshold for speech detection
TEMP_OUTPUT_WAV = "temp_ws_output.wav"

# Load Silero VAD model
torch.hub.set_dir("./.torch_hub")
MODEL, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    source="github",
    model="silero_vad",
    onnx=False,
    trust_repo=True,
)

# Per-client state (audio processing - keep in memory for performance)
STATE = {}

# Per-client locks for state synchronization
CLIENT_LOCKS = {}

# Redis connection pool for better performance
class RedisManager:
    def __init__(self):
        self.pool = None
        self._lock = asyncio.Lock()
    
    async def get_pool(self):
        if self.pool is None:
            async with self._lock:
                if self.pool is None:
                    self.pool = aioredis.ConnectionPool.from_url(
                        REDIS_URL, 
                        decode_responses=True,
                        max_connections=60,
                        retry_on_timeout=True
                    )
        return aioredis.Redis(connection_pool=self.pool)
    
    async def close(self):
        if self.pool:
            await self.pool.disconnect()

redis_manager = RedisManager()

def get_client_lock(client_id):
    """Get or create a lock for the client"""
    if client_id not in CLIENT_LOCKS:
        CLIENT_LOCKS[client_id] = asyncio.Lock()
    return CLIENT_LOCKS[client_id]

def get_state(client_id):
    return STATE.get(client_id, {
        "buffer": [],
        "silence_frames": 0,
        "speech_triggered": False,
        "speech_start": None,
        "speech_duration": 0.0,
        "last_prediction_at": 0.0,
    })

def save_state(client_id, st):
    STATE[client_id] = st

async def get_interview_state(client_id):
    """Get interview state from Redis with synchronization"""
    lock = get_client_lock(client_id)
    async with lock:
        try:
            redis_client = await redis_manager.get_pool()
            state_data = await redis_client.hgetall(f"interview_state:{client_id}")
            if state_data:
                return {
                    "question_idx": int(state_data.get("question_idx", 0)),
                    "phase": state_data.get("phase", "greeting"),
                    "open_chat_turns": int(state_data.get("open_chat_turns", 0)),
                    "last_question_context": state_data.get("last_question_context", ""),
                    "done": state_data.get("done", "false").lower() == "true",
                    "follow_up_count": int(state_data.get("follow_up_count", 0)),
                    "conversation_history": state_data.get("conversation_history", ""),
                    "last_question_asked": state_data.get("last_question_asked", ""),  # Critical for repeat functionality
                    "performance_scores": json.loads(state_data.get("performance_scores", "[]")),
                    "average_score": float(state_data.get("average_score", 5.0)),
                    "role_title": state_data.get("role_title", "Software Developer"),  # Include role
                    "years_experience": state_data.get("years_experience", "2-5 years"),  # Include experience
                    "candidate_name": state_data.get("candidate_name", "Candidate"),  # Include candidate name
                    "company_name": state_data.get("company_name", "Company")  # Include company name
                }
            else:
                # Return default state for new interviews
                return {
                    "question_idx": 0,
                    "phase": "greeting",
                    "open_chat_turns": 0,
                    "last_question_context": "",
                    "done": False,
                    "follow_up_count": 0,
                    "conversation_history": "",
                    "last_question_asked": "",  # Critical for repeat functionality
                    "performance_scores": [],
                    "average_score": 5.0,
                    "role_title": "Software Developer",  # Default role
                    "years_experience": "2-5 years",  # Default experience
                    "candidate_name": "Candidate",  # Default candidate name
                    "company_name": "Company"  # Default company name
                }
        except Exception as e:
            pass
            return {
                "question_idx": 0,
                "phase": "greeting",
                "open_chat_turns": 0,
                "last_question_context": "",
                "done": False,
                "follow_up_count": 0,
                "conversation_history": "",
                "last_question_asked": "",  # Critical for repeat functionality
                "performance_scores": [],
                "average_score": 5.0,
                "role_title": "Software Developer",  # Default role
                "years_experience": "2-5 years"  # Default experience
            }

async def save_interview_state(client_id, state):
    """Save interview state to Redis with synchronization"""
    lock = get_client_lock(client_id)
    async with lock:
        try:
            redis_client = await redis_manager.get_pool()
            await redis_client.hset(f"interview_state:{client_id}", mapping={
                "question_idx": str(state.get("question_idx", 0)),
                "phase": state.get("phase", "greeting"),
                "open_chat_turns": str(state.get("open_chat_turns", 0)),
                "last_question_context": state.get("last_question_context", ""),
                "done": str(state.get("done", False)).lower(),
                "follow_up_count": str(state.get("follow_up_count", 0)),
                "conversation_history": state.get("conversation_history", ""),
                "last_question_asked": state.get("last_question_asked", ""),  # Critical for repeat functionality
                "performance_scores": json.dumps(state.get("performance_scores", [])),
                "average_score": str(state.get("average_score", 5.0)),
                "role_title": state.get("role_title", "Software Developer"),  # Save role
                "years_experience": state.get("years_experience", "2-5 years"),  # Save experience
                "candidate_name": state.get("candidate_name", "Candidate"),  # Save candidate name
                "company_name": state.get("company_name", "Company")  # Save company name
            })
            # Set expiration to 24 hours (UTC-based)
            await redis_client.expire(f"interview_state:{client_id}", 86400)
        except Exception as e:
            pass

async def clear_audio_state(client_id):
    """Clear only audio processing state (not interview state)"""
    if client_id in STATE:
        del STATE[client_id]

async def clear_interview_state(client_id):
    """Clear interview state from Redis"""
    lock = get_client_lock(client_id)
    async with lock:
        try:
            redis_client = await redis_manager.get_pool()
            await redis_client.delete(f"interview_state:{client_id}")
        except Exception as e:
            pass

async def clear_all_state(client_id):
    """Clear both audio and interview state - use only on disconnect"""
    lock = get_client_lock(client_id)
    async with lock:
        # Clear audio state
        if client_id in STATE:
            del STATE[client_id]
        
        # Clear interview state from Redis
        try:
            redis_client = await redis_manager.get_pool()
            await redis_client.delete(f"interview_state:{client_id}")
            # Force close Redis connection to free up pool
            await redis_client.close()
        except Exception as e:
            pass
        
        # Clean up the lock itself
        if client_id in CLIENT_LOCKS:
            del CLIENT_LOCKS[client_id]

async def poll_and_update_interview_state(task_id, client_id):
    """Poll Celery task, send to WebSocket and update interview state when complete"""
    from ai_interview.celery_app import celery_app
    
    try:
        # Poll for task completion directly
        task_result = celery_app.AsyncResult(task_id)
        start_time = asyncio.get_event_loop().time()
        timeout = 60  # Reduced timeout for faster recovery
        
        while not task_result.ready():
            if asyncio.get_event_loop().time() - start_time > timeout:
                await manager.send_to_room(client_id, json.dumps({
                    "error": f"Task {task_id} timed out after {timeout}s",
                    "task_type": "natural_interview"
                }))
                return
            await asyncio.sleep(0.1)
        
        # Task completed, get result
        if task_result.successful():
            result = task_result.result
            print(f"🏁 [WEBSOCKET] Task {task_id} completed successfully")
            print(f"🏁 [WEBSOCKET] Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            print(f"🏁 [WEBSOCKET] Result interview_status: {result.get('interview_status', 'NOT_FOUND') if isinstance(result, dict) else 'N/A'}")
            
            # Update interview state in Redis based on result
            if result and isinstance(result, dict):
                interview_state = await get_interview_state(client_id)
                # Update question index if provided
                if 'question_idx' in result:
                    interview_state['question_idx'] = result['question_idx']
                
                # Update phase if provided
                if 'phase' in result:
                    interview_state['phase'] = result['phase']
                
                # Update follow-up count if provided
                if 'follow_up_count' in result:
                    interview_state['follow_up_count'] = result['follow_up_count']
                
                # Update conversation history if provided
                if 'conversation_history' in result:
                    interview_state['conversation_history'] = result['conversation_history']
                
                # Update completion status
                if 'done' in result:
                    interview_state['done'] = result['done']
                    
                    # If interview is completed, clear the generated content cache
                    if result['done']:
                        room_id = client_id  # room_id is same as client_id in WebSocket
                        print(f"🧹 Interview completed for room {room_id} - clearing generated cache")
                        # clear_room_generated_cache(room_id)
                
                # Update last question asked - CRITICAL for repeat functionality
                if 'last_question_asked' in result:
                    interview_state['last_question_asked'] = result['last_question_asked']
                
                # Update performance tracking
                if 'performance_scores' in result:
                    interview_state['performance_scores'] = result['performance_scores']
                
                if 'average_score' in result:
                    interview_state['average_score'] = result['average_score']
                    
                # Save updated state to Redis
                await save_interview_state(client_id, interview_state)
            
            # Send result to WebSocket
            response = {
                "task_id": task_id,
                "task_type": "natural_interview",
                "status": "completed",
                "result": result
            }
            await manager.send_to_room(client_id, json.dumps(response))
        else:
            await manager.send_to_room(client_id, json.dumps({
                "task_id": task_id,
                "task_type": "natural_interview",
                "status": "failed",
                "error": str(task_result.result)
            }))
            
    except Exception as e:
        pass
        await manager.send_to_room(
            client_id,
            json.dumps({
                "task_type": "natural_interview",
                "status": "error",
                "error": str(e)
            })
        )

def process_segment(audio_buffer, speech_start_time):
    """Simple silence-based turn detection - no ML model"""
    # Always return Complete since we're using silence duration for turn detection
    return {
        "prediction": "Complete",
        "probability": 1.0
    }

def decode_audio_base64(b64_str):
    """Decode base64 PCM16 WAV chunk - EXACT MATCH with record_and_predict.py"""
    pcm = base64.b64decode(b64_str.split(",", 1)[-1])
    arr16 = np.frombuffer(pcm, dtype=np.int16)
    return arr16.astype(np.float32) / np.iinfo(np.int16).max  # EXACT MATCH


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    
    # Register WebSocket with manager
    await manager.connect(client_id, websocket)

    state = get_state(client_id)

    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            msg_type = payload.get("type")

            # --- Audio handling ---
            if msg_type == "audio":
                b64 = payload.get("audio") or payload.get("text")
                if not b64:
                    await websocket.send_json({"error": "Missing audio field"})
                    continue
                # decode PCM16 -> float32
                audio_float32 = decode_audio_base64(b64)
                now = time.time()

                # VAD PROCESSING - Early exit on first speech detection
                is_speech = False
                all_probs = []
                
                for start in range(0, len(audio_float32), CHUNK):
                    chunk = audio_float32[start:start+CHUNK]
                    if len(chunk) < CHUNK:
                        continue
                    prob = MODEL(torch.from_numpy(chunk).unsqueeze(0), RATE).item()
                    all_probs.append(prob)
                    if prob > VAD_THRESHOLD:
                        is_speech = True
                        break  # Early exit on first speech detection
                
                # Enhanced speech detection for continuous speech
                if not is_speech:
                    max_prob = max(all_probs) if all_probs else 0.0
                    if max_prob > 0.3:
                        is_speech = True

                # SPEECH BRANCH
                if is_speech:
                    # 1. Set speech_triggered = True AND reset silence counter
                    state["silence_frames"] = 0  # Always reset when speech detected
                    if not state.get("speech_triggered"):
                        print(f"[{client_id}] 🎤 SPEECH STARTED at {now:.3f}")
                        state["speech_start_timestamp"] = now  # Track speech start time
                    else:
                        silence_duration = state.get("silence_frames", 0) * (CHUNK / RATE)
                        print(f"[{client_id}] 🔄 SPEECH RESUMED after {silence_duration:.2f}s pause - silence counter reset")
                    state["speech_triggered"] = True
                    
                    # 2. Add audio to buffer
                    state["buffer"].append((now, audio_float32))
                    
                    # 3. Update speech_duration
                    if state.get("speech_start") is None:
                        state["speech_start"] = now
                    state["speech_duration"] = now - state["speech_start"]
                    
                    save_state(client_id, state)
                
                # SILENCE BRANCH
                else:
                    if state.get("speech_triggered"):
                        # 1. Add to buffer (continue buffering during silence)
                        state["buffer"].append((now, audio_float32))
                        
                        # 2. Increment silence_frames and track timing
                        if state.get("silence_frames") == 0:
                            state["silence_start_time"] = now  # Track when silence began
                        state["silence_frames"] = state.get("silence_frames", 0) + 1
                        
                        # 3. Calculate silence_duration
                        silence_duration = state["silence_frames"] * (CHUNK / RATE)
                        
                        # Debug silence progress every 0.5s
                        if int(silence_duration * 2) % 1 == 0 and silence_duration <= 3.0:
                            print(f"[{client_id}] 🔇 Silence: {silence_duration:.1f}s / 2.0s")
                        
                        # SILENCE >= 2.0 SECONDS?
                        if silence_duration >= 3.0:
                            
                            # Calculate total response time metrics
                            total_speech_time = now - state.get("speech_start_timestamp", now)
                            pause_detection_time = now - state.get("silence_start_time", now)
                            
                            # END TURN based on silence duration only
                            result = process_segment(state["buffer"], state["speech_start"])
                            
                            print(f"[{client_id}] ✅ TURN ENDED - 2s silence detected:")
                            print(f"    📊 Total speech: {total_speech_time:.1f}s")
                            print(f"    ⏱️  Pause detection: {pause_detection_time:.2f}s")
                            print(f"    ✅ Result: {result['prediction']} (prob: {result['probability']:.3f})")
                            
                            await clear_audio_state(client_id)
                            await websocket.send_json({"status": "segment", **result})
                            state = get_state(client_id)
                        
                        save_state(client_id, state)
                    else:
                        # Pre-speech buffering
                        state["buffer"].append((now, audio_float32))
                        max_buffer_time = (PRE_SPEECH_MS + STOP_MS) / 1000.0 + MAX_DURATION_SECONDS
                        state["buffer"] = [(t, c) for t, c in state["buffer"] if t >= now - max_buffer_time]
                        save_state(client_id, state)
                
                save_state(client_id, state)

            # --- Video handling ---
            elif msg_type == "video":
                payload["room_id"] = client_id
                task = process_video.delay(payload)
                await manager.send_to_room(
                    client_id,
                    json.dumps({"task_id": task.id, "task_type": "video"})
                )

            # # --- Restart interview (clear state and start fresh) with role support ---
            # elif msg_type == "restart_interview":
            #     # Clear existing interview state for fresh start
            #     await clear_interview_state(client_id)
                
            #     # Get fresh interview state from Redis (will be default state)
            #     interview_state = await get_interview_state(client_id)
                
            #     # Extract role and experience from payload if provided
            #     role_title = payload.get("role_title", "Full Stack Developer")  # Default fallback
            #     years_experience = payload.get("years_experience", "2-5 years")  # Default fallback
                
            #     print(f"[{client_id}] 🔄 Restarting interview for role: {role_title}, experience: {years_experience}")
                
            #     # Create payload without user input to trigger greeting
            #     payload = {
            #         "room_id": client_id,
            #         "text": "",  # No user input for initial greeting
            #         "role_title": role_title,  # Pass role to natural interview flow
            #         "years_experience": years_experience,  # Pass experience to natural interview flow
            #         **interview_state
            #     }
                
            #     task = process_natural_interview.delay(payload)
                
            #     # Poll and send result to WebSocket, then update interview state
            #     asyncio.create_task(
            #         poll_and_update_interview_state(task.id, client_id)
            #     )


            # --- Start interview (greeting) handling with role support ---
            # elif msg_type == "start_interview":
            #     # Get current interview state from Redis (persistent across reconnects)
            #     interview_state = await get_interview_state(client_id)
                
            #     # Extract role and experience from payload - frontend always takes precedence
            #     role_title = payload.get("role_title") or payload.get("role_type") or interview_state.get("role_title", "Software Developer")
            #     years_experience = payload.get("years_experience") or interview_state.get("years_experience", "2-5 years")
            #     candidate_name = payload.get("candidate_name", interview_state.get("candidate_name", "Candidate"))
            #     company_name = payload.get("company_name", interview_state.get("company_name", "Company"))
                
            #     # Create payload without user input to trigger greeting
            #     payload = {
            #         **interview_state,  # Add all state fields first
            #         "room_id": client_id,
            #         "text": "",  # No user input for initial greeting
            #         "role_title": role_title,  # Use role_title from frontend for natural interview flow (this overrides state)
            #         "years_experience": years_experience,  # Pass experience to natural interview flow (this overrides state)
            #         "candidate_name": candidate_name,  # Pass candidate name to natural interview flow (this overrides state)
            #         "company_name": company_name,  # Pass company name to natural interview flow (this overrides state)
            #     }
                
            #     task = process_natural_interview.delay(payload)
                
            #     # Poll and send result to WebSocket, then update interview state
            #     asyncio.create_task(
            #         poll_and_update_interview_state(task.id, client_id)
            #     )

            # --- Natural interview (answer) handling with role support ---
            elif msg_type == "answer":
                
                # Get current interview state from Redis and merge with payload
                interview_state = await get_interview_state(client_id)
                
                # Create proper payload structure - preserve user's text input
                user_text = payload.get("text", "")
                # Extract role and experience from payload - frontend always takes precedence  
                role_title = payload.get("role_title") or payload.get("role_type") or interview_state.get("role_title", "Software Developer")
                years_experience = payload.get("years_experience") or interview_state.get("years_experience", "2-5 years")
                candidate_name = payload.get("candidate_name", interview_state.get("candidate_name", "Candidate"))
                company_name = payload.get("company_name", interview_state.get("company_name", "Company"))
                
                # Start background generation for skills/questions if not already started
                from ai_interview.services.interview_flow_service import start_background_generation
                start_background_generation(role_title, years_experience, client_id)
                
                # Ensure non-empty text for proper routing
                if not user_text.strip():
                    pass
                
                merged_payload = {
                    **interview_state,  # Add all state fields first
                    "room_id": client_id,
                    "text": user_text,  # Always preserve user's actual input
                    "role_title": role_title,  # Use role_title from frontend for natural interview flow (this overrides state)
                    "years_experience": years_experience,  # Pass experience to natural interview flow (this overrides state)
                    "candidate_name": candidate_name,  # Pass candidate name to natural interview flow (this overrides state)
                    "company_name": company_name,  # Pass company name to natural interview flow (this overrides state)
                }
                
                # Process interview flow directly using the service
                try:
                    result = await interview_flow_service.process_interview_flow(merged_payload)
                    
                    # Update interview state in Redis based on result
                    if result and isinstance(result, dict):
                        interview_state = await get_interview_state(client_id)
                        
                        # Update question index if provided
                        if 'question_idx' in result:
                            interview_state['question_idx'] = result['question_idx']
                        
                        # Update phase if provided
                        if 'phase' in result:
                            interview_state['phase'] = result['phase']
                        
                        # Update follow-up count if provided
                        if 'follow_up_count' in result:
                            interview_state['follow_up_count'] = result['follow_up_count']
                        
                        # Update conversation history if provided
                        if 'conversation_history' in result:
                            interview_state['conversation_history'] = result['conversation_history']
                        
                        # Update completion status
                        if 'done' in result:
                            interview_state['done'] = result['done']
                        
                        # Update last question asked - CRITICAL for repeat functionality
                        if 'last_question_asked' in result:
                            interview_state['last_question_asked'] = result['last_question_asked']
                        
                        # Save updated state to Redis
                        await save_interview_state(client_id, interview_state)
                    
                    # Send result to WebSocket
                    response = {
                        "task_type": "natural_interview",
                        "status": "completed",
                        "result": result
                    }
                    await manager.send_to_room(client_id, json.dumps(response))
                    
                except Exception as e:
                    logger.error(f"Error processing interview flow: {e}")
                    await manager.send_to_room(client_id, json.dumps({
                        "task_type": "natural_interview",
                        "status": "error",
                        "error": str(e)
                    }))

            else:
                await websocket.send_json({"error": f"Unsupported message type: {msg_type}"})

    except WebSocketDisconnect:
        pass
        # Clean up Celery tasks for this client
        try:
            # Get all active tasks for this client and revoke them
            active_tasks = celery_app.control.inspect().active()
            if active_tasks:
                for worker, tasks in active_tasks.items():
                    for task in tasks:
                        if task.get('args') and len(task['args']) > 0:
                            # Check if task is for this client (room_id in payload)
                            task_payload = task['args'][0] if task['args'] else {}
                            if isinstance(task_payload, dict) and task_payload.get('room_id') == client_id:
                                print(f"[CLEANUP] Revoking Celery task {task['id']} for client {client_id}")
                                celery_app.control.revoke(task['id'], terminate=True)
            
            # Also revoke any scheduled tasks
            scheduled_tasks = celery_app.control.inspect().scheduled()
            if scheduled_tasks:
                for worker, tasks in scheduled_tasks.items():
                    for task in tasks:
                        if task.get('request', {}).get('args') and len(task['request']['args']) > 0:
                            task_payload = task['request']['args'][0]
                            if isinstance(task_payload, dict) and task_payload.get('room_id') == client_id:
                                print(f"[CLEANUP] Revoking scheduled Celery task {task['request']['id']} for client {client_id}")
                                celery_app.control.revoke(task['request']['id'], terminate=True)
        except Exception as e:
            print(f"[ERROR] Failed to revoke Celery tasks for {client_id}: {e}")
        
        await manager.disconnect(client_id, websocket)
        await clear_audio_state(client_id)  # Only clear audio state, keep interview state persistent
        
        # Clear generated content cache on disconnect to free up memory
        room_id = client_id  # room_id is same as client_id in WebSocket
        print(f"🧹 Room {room_id} disconnected - clearing generated cache")
        # clear_room_generated_cache(room_id)
        
        # Clear session cache for skills/questions
        from ai_interview.services.interview_flow_service import clear_session_cache
        clear_session_cache(client_id)
