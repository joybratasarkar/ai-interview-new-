# apps/ai_interview/utils/redis_state.py

import redis
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

from ai_interview.config import REDIS_URL

r = redis.Redis.from_url(REDIS_URL)

def set_state(room_id: str, state: dict, ttl: int = 3600) -> None:
    """Save session state with TTL (default 1 hour)"""
    print(f"[SET_STATE] Saving state for {room_id}: {state}")
    r.setex(f"interview:session:{room_id}", ttl, json.dumps(state))

def get_state(room_id: str) -> dict:
    """Get session state"""
    data = r.get(f"interview:session:{room_id}")
    if data:
        state = json.loads(data)
        print(f"[GET_STATE] Loaded state for {room_id}: {state}")
        return state
    print(f"[GET_STATE] No state found for {room_id}")
    return None

def save_conversation_turn(room_id: str, user_message: str, bot_response: str) -> None:
    """Store conversation turn in Redis list with timestamp"""
    turn_data = {
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message,
        "bot_response": bot_response
    }
    # Store in Redis list with TTL
    conversation_key = f"interview:conversation:{room_id}"
    r.lpush(conversation_key, json.dumps(turn_data))
    r.expire(conversation_key, 86400)  # 24 hours TTL

def get_conversation_history(room_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get conversation history for room_id"""
    conversation_key = f"interview:conversation:{room_id}"
    conversation_data = r.lrange(conversation_key, 0, limit - 1)
    
    history = []
    for turn_json in reversed(conversation_data):  # Reverse to get chronological order
        try:
            turn = json.loads(turn_json)
            history.append(turn)
        except json.JSONDecodeError:
            continue
    
    return history

def clear_conversation(room_id: str) -> None:
    """Clear conversation history for room_id"""
    conversation_key = f"interview:conversation:{room_id}"
    r.delete(conversation_key)

def acquire_lock(room_id: str, timeout: int = 5) -> bool:
    return r.set(f"interview:lock:{room_id}", "1", nx=True, ex=timeout)

def release_lock(room_id: str):
    r.delete(f"interview:lock:{room_id}")
