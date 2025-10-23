import asyncio
import redis.asyncio as aioredis
from ai_interview.config import REDIS_URL

# Global Redis client (async)
redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)

# In-memory stream states (shared dict)
stream_states = {}

async def cleanup_audio_session(room_id: str):
    # 1. Remove stream buffer
    if room_id in stream_states:
        del stream_states[room_id]
        print(f"[CLEANUP] Cleared stream state for room: {room_id}")

    # 2. Delete Redis interview-ended flag
    try:
        await redis_client.delete(f"interview:ended:{room_id}")
        print(f"[CLEANUP] Deleted Redis key: interview:ended:{room_id}")
    except Exception as e:
        print(f"[CLEANUP ERROR] Could not delete Redis key for {room_id}: {e}")

    # 3. Delete lock if still exists
    try:
        await redis_client.delete(f"lock:{room_id}")
        print(f"[CLEANUP] Deleted lock for room: {room_id}")
    except Exception as e:
        print(f"[CLEANUP ERROR] Could not delete lock for {room_id}: {e}")

    # 4. Optionally delete any stream data key
    try:
        await redis_client.delete(f"stream:{room_id}")
        print(f"[CLEANUP] Deleted Redis key: stream:{room_id}")
    except Exception as e:
        print(f"[CLEANUP ERROR] Could not delete stream key for {room_id}: {e}")
