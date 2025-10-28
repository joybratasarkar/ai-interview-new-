from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
import base64
import time
import torch
import numpy as np
from scipy.io import wavfile
from ai_interview.smart_turn.inference import predict_endpoint
from ai_interview.celery_app import celery_app
from ai_interview.tasks.video import process_video
from ai_interview.tasks.natural_interview_flow import (
    process_natural_interview,
)
from ai_interview.services.socket_manager import manager
import redis.asyncio as aioredis
from ai_interview.config import REDIS_URL

router = APIRouter()

# --- Config --- OPTIMIZED FOR FASTER TURN DETECTION
`RATE = 16000
CHUNK = 512  # Standard chunk size for stable audio processing
STOP_MS = 600  # Further reduced for faster turn detection
PRE_SPEECH_MS = 150  # Reduced pre-speech buffer
MAX_DURATION_SECONDS = 16
VAD_THRESHOLD = 0.5  # Lower threshold for more sensitive/faster detection
TEMP_OUTPUT_WAV = "temp_ws_output.wav"`

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
                        max_connections=20,
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
        "speech_start": None
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
                    "conversation_history": state_data.get("conversation_history", "")
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
                    "conversation_history": ""
                }
        except Exception as e:
            print(f"Error getting interview state from Redis: {e}")
            return {
                "question_idx": 0,
                "phase": "greeting",
                "open_chat_turns": 0,
                "last_question_context": "",
                "done": False,
                "follow_up_count": 0,
                "conversation_history": ""
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
                "conversation_history": state.get("conversation_history", "")
            })
            # Set expiration to 24 hours
            await redis_client.expire(f"interview_state:{client_id}", 86400)
            print(f"Saved interview state to Redis for {client_id}: {state}")
        except Exception as e:
            print(f"Error saving interview state to Redis: {e}")

async def clear_audio_state(client_id):
    """Clear only audio processing state (not interview state)"""
    if client_id in STATE:
        del STATE[client_id]
        print(f"Cleared audio state for {client_id}")

async def clear_interview_state(client_id):
    """Clear interview state from Redis"""
    lock = get_client_lock(client_id)
    async with lock:
        try:
            redis_client = await redis_manager.get_pool()
            await redis_client.delete(f"interview_state:{client_id}")
            print(f"Cleared interview state from Redis for {client_id}")
        except Exception as e:
            print(f"Error clearing interview state from Redis: {e}")

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
            print(f"Cleared all state for {client_id}")
        except Exception as e:
            print(f"Error clearing all state: {e}")
        
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
        timeout = 30
        
        while not task_result.ready():
            if asyncio.get_event_loop().time() - start_time > timeout:
                print(f"Task {task_id} timed out after {timeout}s")
                await manager.send_to_room(client_id, json.dumps({
                    "error": f"Task {task_id} timed out after {timeout}s",
                    "task_type": "natural_interview"
                }))
                return
            await asyncio.sleep(0.1)
        
        print(f"Task {task_id} completed with status: {task_result.status}")
        
        # Task completed, get result
        if task_result.successful():
            result = task_result.result
            print(f"Interview task result: {result}")
            
            # Update interview state in Redis based on result
            if result and isinstance(result, dict):
                interview_state = await get_interview_state(client_id)
                print(f"Current interview state from Redis: {interview_state}")
                # Update question index if provided
                if 'question_idx' in result:
                    old_idx = interview_state.get('question_idx', 0)
                    interview_state['question_idx'] = result['question_idx']
                    print(f"🔄 QUESTION INDEX UPDATE: {old_idx} → {result['question_idx']} (Phase: {result.get('phase', 'unknown')})")
                
                # Update phase if provided
                if 'phase' in result:
                    interview_state['phase'] = result['phase']
                    print(f"Updating Redis phase to: {result['phase']}")
                
                # Update follow-up count if provided
                if 'follow_up_count' in result:
                    interview_state['follow_up_count'] = result['follow_up_count']
                    print(f"Updating Redis follow_up_count to: {result['follow_up_count']}")
                
                # Update conversation history if provided
                if 'conversation_history' in result:
                    interview_state['conversation_history'] = result['conversation_history']
                    print(f"Updating Redis conversation_history")
                
                # Update completion status
                if 'done' in result:
                    interview_state['done'] = result['done']
                    
                # Save updated state to Redis
                await save_interview_state(client_id, interview_state)
                print(f"Updated interview state in Redis for client {client_id}: {interview_state}")
            
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
        print(f"Error polling task {task_id} for client {client_id}: {e}")
        await manager.send_to_room(
            client_id,
            json.dumps({
                "task_type": "natural_interview",
                "status": "error",
                "error": str(e)
            })
        )

def process_segment(audio_buffer, speech_start_time):
    """OPTIMIZED: Process audio segment with faster processing"""
    start_time = speech_start_time - (PRE_SPEECH_MS / 1000)
    start_index = 0
    for i, (t, _) in enumerate(audio_buffer):
        if t >= start_time:
            start_index = i
            break

    segment_audio_chunks = [chunk for _, chunk in audio_buffer[start_index:]]
    segment_audio = np.concatenate(segment_audio_chunks)

    # OPTIMIZED: Remove minimal trailing silence for fastest response
    samples_to_remove = int(50 / 1000 * RATE)  # Further reduced to 50ms
    if len(segment_audio) > samples_to_remove:
        segment_audio = segment_audio[:-samples_to_remove]

    # limit max duration
    if len(segment_audio) / RATE > MAX_DURATION_SECONDS:
        segment_audio = segment_audio[: int(MAX_DURATION_SECONDS * RATE)]

    # OPTIMIZED: Skip file writing in production for speed
    # Only write for debugging when needed
    # wavfile.write(TEMP_OUTPUT_WAV, RATE, (segment_audio * 32767).astype(np.int16))

    # predict - this is the main bottleneck (50-150ms)
    result = predict_endpoint(segment_audio)
    label = "Complete" if result["prediction"] == 1 else "Incomplete"
    return {
        "prediction": label,
        "probability": result["probability"]
    }

def decode_audio_base64(b64_str, target_rate=RATE):
    """Decode base64 PCM16 WAV chunk."""
    pcm = base64.b64decode(b64_str.split(",", 1)[-1])
    arr16 = np.frombuffer(pcm, dtype=np.int16)
    return arr16.astype(np.float32) / 32767.0

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    print(f"[WS] Connected: {client_id}")
    
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

                # OPTIMIZED VAD: Early exit for faster detection (like previous.py)
                is_speech = False
                for start in range(0, len(audio_float32), CHUNK):
                    chunk = audio_float32[start:start+CHUNK]
                    if len(chunk) < CHUNK:
                        continue
                    prob = MODEL(torch.from_numpy(chunk).unsqueeze(0), RATE).item()
                    if prob > VAD_THRESHOLD:
                        is_speech = True
                        break  # Early exit on first speech detection - FASTER

                # buffer logic
                if is_speech:
                    if not state["speech_triggered"]:
                        state["silence_frames"] = 0
                    state["speech_triggered"] = True
                    if state["speech_start"] is None:
                        state["speech_start"] = now
                    state["buffer"].append((now, audio_float32))
                    save_state(client_id, state)
                else:
                    if state["speech_triggered"]:
                        state["buffer"].append((now, audio_float32))
                        state["silence_frames"] += 1
                        if state["silence_frames"] * (CHUNK / RATE) >= STOP_MS / 1000:
                            result = process_segment(state["buffer"], state["speech_start"])
                            await clear_audio_state(client_id)  # Only clear audio state, not interview state
                            await websocket.send_json({"status": "segment", **result})
                            state = get_state(client_id)
                        else:
                            save_state(client_id, state)
                    else:
                        state["buffer"].append((now, audio_float32))
                        max_buffer_time = (PRE_SPEECH_MS + STOP_MS) / 1000.0 + MAX_DURATION_SECONDS
                        state["buffer"] = [(t, c) for t, c in state["buffer"] if t >= now - max_buffer_time]
                        save_state(client_id, state)

            # --- Video handling ---
            elif msg_type == "video":
                payload["room_id"] = client_id
                task = process_video.delay(payload)
                await manager.send_to_room(
                    client_id,
                    json.dumps({"task_id": task.id, "task_type": "video"})
                )

            # --- Start interview (greeting) handling ---
            elif msg_type == "start_interview":
                # Get current interview state from Redis
                interview_state = await get_interview_state(client_id)
                
                # Create payload without user input to trigger greeting
                payload = {
                    "room_id": client_id,
                    "text": "",  # No user input for initial greeting
                    **interview_state
                }
                
                print(f"Starting interview for client {client_id}")
                print(f"Initial interview state: {interview_state}")
                
                task = process_natural_interview.delay(payload)
                print(f"Started greeting task: {task.id} for client: {client_id}")
                
                # Poll and send result to WebSocket, then update interview state
                asyncio.create_task(
                    poll_and_update_interview_state(task.id, client_id)
                )

            # --- Natural interview (answer) handling ---
            elif msg_type == "answer":
                print('🔍 WEBSOCKET ANSWER PROCESSING START')
                print('--------------------------------------------------')
                
                # Get current interview state from Redis and merge with payload
                interview_state = await get_interview_state(client_id)
                
                # Create proper payload structure - preserve user's text input
                user_text = payload.get("text", "")
                
                # FRONTEND DEBUG: Ensure non-empty text for proper routing
                if not user_text.strip():
                    print("⚠️  WARNING: Empty user text - this may cause routing issues")
                
                merged_payload = {
                    "room_id": client_id,
                    "text": user_text,  # Always preserve user's actual input
                    **interview_state  # Add all state fields
                }
                
                print(f"📝 USER INPUT: '{user_text[:50]}...'")
                print(f"📦 REDIS STATE: {interview_state}")
                print(f"🚀 PAYLOAD TO CELERY: {merged_payload}")
                print(f"🎯 KEY ROUTING INFO: phase={interview_state.get('phase')}, question_idx={interview_state.get('question_idx')}, user_input_empty={not user_text.strip()}")
                
                
                task = process_natural_interview.delay(merged_payload)
                print(f"Started natural interview task: {task.id} for client: {client_id}")
                
                # Poll and send result to WebSocket, then update interview state
                asyncio.create_task(
                    poll_and_update_interview_state(task.id, client_id)
                )

            else:
                await websocket.send_json({"error": f"Unsupported message type: {msg_type}"})

    except WebSocketDisconnect:
        print(f"[WS] Disconnected: {client_id}")
        await manager.disconnect(client_id, websocket)
        await clear_all_state(client_id)  # Clear everything on disconnect
