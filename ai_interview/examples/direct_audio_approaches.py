# Example approaches for sending audio results directly without pub/sub

"""
OPTION 1: Return Result from Celery Task (Recommended)
=====================================================
Modify audio.py to return result instead of publishing to pub/sub
"""

# In audio.py - Modified _submit_and_cleanup function
def _submit_and_cleanup_v2(room_id: str) -> dict:
    """Return result instead of publishing to pub/sub"""
    logger.info(f"Final pause submitted for room '{room_id}'")
    
    # Clear in-memory state
    stream_states.pop(room_id, None)
    paused_rooms[room_id] = datetime.utcnow()
    
    # Mark turn-ended in Redis (TTL 5 minutes)
    dis.setex(f"turn-ended:{room_id}", 300, "1")
    
    # Return result instead of publishing
    return {
        "room_id": room_id, 
        "task_type": "audio", 
        "result": True,
        "timestamp": datetime.utcnow().isoformat()
    }

# In audio.py - Modified process_audio task
@celery_app.task(name="ai_interview.tasks.audio.process_audio_v2")
def process_audio_v2(payload):
    # ... existing audio processing logic ...
    
    while len(st.buffer) >= CHUNK_SIZE:
        window = np.array([st.buffer.popleft() for _ in range(CHUNK_SIZE)], dtype=np.float32)
        
        if st.process_chunk(window):
            logger.info(f"Tentative pause detected in room '{room_id}'. Waiting {DEBOUNCE_DELAY}s for confirmation.")
            time.sleep(DEBOUNCE_DELAY)
            
            silent = st.process_chunk(np.zeros(CHUNK_SIZE, dtype=np.float32))
            if silent:
                logger.info(f"✅ Confirmed pause in room '{room_id}'. Submitting event.")
                # Return result instead of pub/sub
                return _submit_and_cleanup_v2(room_id)
            else:
                logger.info(f"❌ False pause detection in room '{room_id}', speech resumed.")
                break
    
    # Return None if no pause detected
    return None


"""
OPTION 2: Direct WebSocket Sending from Celery
==============================================
Send directly to WebSocket manager from Celery task
"""

# In audio.py - Import WebSocket manager
from ai_interview.services.shared import manager

# Modified _submit_and_cleanup with direct WebSocket sending
async def _submit_and_cleanup_direct(room_id: str):
    """Send result directly to WebSocket"""
    logger.info(f"Final pause submitted for room '{room_id}'")
    
    # Clear state
    stream_states.pop(room_id, None)
    paused_rooms[room_id] = datetime.utcnow()
    dis.setex(f"turn-ended:{room_id}", 300, "1")
    
    # Send directly to WebSocket
    result = {
        "room_id": room_id,
        "task_type": "audio", 
        "result": True,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Direct WebSocket send
    await manager.send_to_room(room_id, json.dumps(result))
    
    # Async cleanup
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_cleanup_room(room_id))
    except RuntimeError:
        asyncio.run(_cleanup_room(room_id))


"""
WEBSOCKET HANDLER CHANGES
========================
"""

# Option 1: Using Celery task result
async def websocket_handler_option1():
    if msg_type == "audio":
        # Get task result directly
        task = process_audio_v2.delay({
            "type": "audio",
            "room_id": client_id,
            "text": text
        })
        
        # Optional: Wait for result (blocking) or poll later
        try:
            result = task.get(timeout=10)  # Wait up to 10 seconds
            if result:
                await manager.send_to_self(
                    client_id, websocket,
                    json.dumps(result)
                )
        except Exception as e:
            logger.error(f"Audio task failed for {client_id}: {e}")

# Option 2: Fire and forget (existing approach but without pub/sub)
async def websocket_handler_option2():
    if msg_type == "audio":
        # Just fire the task - it will send directly to WebSocket
        process_audio_direct.delay({
            "type": "audio",
            "room_id": client_id,
            "text": text
        })


"""
RECOMMENDATION: Use Option 1 with async task result polling
==========================================================
"""

# Best approach: Non-blocking task result checking
import asyncio
from celery.result import AsyncResult

async def websocket_handler_recommended():
    if msg_type == "audio":
        task = process_audio_v2.delay({
            "type": "audio", 
            "room_id": client_id,
            "text": text
        })
        
        # Send task_id immediately
        await manager.send_to_self(
            client_id, websocket,
            json.dumps({"task_id": task.id, "task_type": "audio", "status": "processing"})
        )
        
        # Start background polling for result
        asyncio.create_task(poll_audio_result(task.id, client_id))

async def poll_audio_result(task_id: str, client_id: str):
    """Poll for audio task result and send when ready"""
    result = AsyncResult(task_id, app=celery_app)
    
    # Poll until ready (with timeout)
    for _ in range(50):  # 5 second timeout (50 * 0.1)
        if result.ready():
            if result.successful():
                audio_result = result.result
                if audio_result:  # Pause detected
                    await manager.send_to_room(
                        client_id, 
                        json.dumps(audio_result)
                    )
            break
        await asyncio.sleep(0.1)