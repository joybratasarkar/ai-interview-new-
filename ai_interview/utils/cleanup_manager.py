# apps/ai_interview/utils/cleanup_manager.py

import json
import asyncio
import os
from pathlib import Path
from typing import Set, List
import redis.asyncio as aioredis
import redis
from celery import current_app as celery_app
from celery.result import AsyncResult

from ai_interview.config import REDIS_URL

class DisconnectCleanupManager:
    """Comprehensive cleanup manager for user disconnection"""
    
    def __init__(self):
        self.redis_sync = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        
    async def cleanup_user_data(self, user_id: str) -> dict:
        """
        Complete cleanup of all user data on disconnect
        Returns: cleanup report
        """
        cleanup_report = {
            "user_id": user_id,
            "celery_tasks_revoked": 0,
            "redis_keys_deleted": 0,
            "pubsub_messages_cleared": 0,
            "temp_files_deleted": 0,
            "errors": []
        }
        
        try:
            # 1. Revoke active Celery tasks
            revoked_count = await self._revoke_user_celery_tasks(user_id)
            cleanup_report["celery_tasks_revoked"] = revoked_count
            
            # 2. Clear Redis user state
            deleted_keys = await self._clear_redis_user_data(user_id)
            cleanup_report["redis_keys_deleted"] = deleted_keys
            
            # 3. Clear pub/sub messages (prevent delivery)
            await self._clear_pubsub_queue(user_id)
            cleanup_report["pubsub_messages_cleared"] = 1
            
            # 4. Clear audio processing state
            await self._clear_audio_state(user_id)
            
            # 5. Delete temporary WAV files
            deleted_files = await self._delete_temp_wav_files(user_id)
            cleanup_report["temp_files_deleted"] = deleted_files
            
            print(f"[CLEANUP] User {user_id} cleanup completed: {cleanup_report}")
            
        except Exception as e:
            cleanup_report["errors"].append(str(e))
            print(f"[CLEANUP ERROR] {user_id}: {e}")
            
        return cleanup_report
    
    async def _revoke_user_celery_tasks(self, user_id: str) -> int:
        """Revoke all pending/active Celery tasks for user"""
        revoked_count = 0
        
        try:
            # Get active tasks from Celery
            inspect = celery_app.control.inspect()
            active_tasks = inspect.active()
            
            if active_tasks:
                for worker, tasks in active_tasks.items():
                    for task in tasks:
                        # Check if task belongs to this user
                        task_args = task.get('args', [])
                        if self._is_user_task(task_args, user_id):
                            task_id = task['id']
                            celery_app.control.revoke(task_id, terminate=True)
                            revoked_count += 1
                            print(f"[REVOKE] Task {task_id} for user {user_id}")
            
            # Also check scheduled tasks
            scheduled = inspect.scheduled()
            if scheduled:
                for worker, tasks in scheduled.items():
                    for task in tasks:
                        task_args = task.get('args', [])
                        if self._is_user_task(task_args, user_id):
                            task_id = task['id']
                            celery_app.control.revoke(task_id, terminate=True)
                            revoked_count += 1
                            
        except Exception as e:
            print(f"[REVOKE ERROR] {user_id}: {e}")
            
        return revoked_count
    
    def _is_user_task(self, task_args: List, user_id: str) -> bool:
        """Check if Celery task belongs to user"""
        try:
            # For audio tasks: args contain payload with room_id
            if task_args and len(task_args) > 0:
                payload = task_args[0]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if isinstance(payload, dict):
                    return payload.get('room_id') == user_id
        except (json.JSONDecodeError, KeyError):
            pass
        return False
    
    async def _clear_redis_user_data(self, user_id: str) -> int:
        """Clear all Redis keys related to user"""
        redis_async = aioredis.from_url(REDIS_URL, decode_responses=True)
        deleted_count = 0
        
        try:
            # Patterns to match user data
            patterns = [
                f"*{user_id}*",           # Any key containing user_id
                f"stream:{user_id}",      # Audio stream data
                f"turn-ended:{user_id}",  # Turn state
                f"interview:{user_id}*",  # Interview state
                f"lock:{user_id}",        # State locks
                f"session:{user_id}*",    # Session data
            ]
            
            for pattern in patterns:
                keys = await redis_async.keys(pattern)
                if keys:
                    deleted = await redis_async.delete(*keys)
                    deleted_count += deleted
                    print(f"[REDIS] Deleted {deleted} keys for pattern: {pattern}")
            
        except Exception as e:
            print(f"[REDIS ERROR] {user_id}: {e}")
        finally:
            await redis_async.close()
            
        return deleted_count
    
    async def _clear_pubsub_queue(self, user_id: str) -> None:
        """Clear any pending pub/sub messages for user"""
        try:
            # Mark user as disconnected to prevent message delivery
            self.redis_sync.setex(f"disconnected:{user_id}", 300, "1")  # 5 min TTL
            
            # Optional: Clear channel if needed (careful with multi-user)
            # This is more aggressive - only use if each user has own channel
            # await redis_async.delete(f"interview_updates:{user_id}")
            
        except Exception as e:
            print(f"[PUBSUB ERROR] {user_id}: {e}")
    
    async def _clear_audio_state(self, user_id: str) -> None:
        """Clear in-memory audio processing state"""
        try:
            # Import here to avoid circular imports
            from ai_interview.tasks.audio import stream_states, paused_rooms
            
            # Clear in-memory state
            stream_states.pop(user_id, None)
            paused_rooms.pop(user_id, None)
            
            print(f"[AUDIO] Cleared in-memory state for {user_id}")
            
        except Exception as e:
            print(f"[AUDIO ERROR] {user_id}: {e}")
    
    async def _delete_temp_wav_files(self, user_id: str) -> int:
        """Delete temporary WAV files for user"""
        deleted_count = 0
        
        try:
            # Common temporary WAV file patterns
            temp_files = [
                "temp_output.wav",
                "temp_ws_output.wav",
                f"temp_{user_id}.wav",
                f"temp_ws_{user_id}.wav",
            ]
            
            # Check current working directory and common paths
            search_paths = [
                ".",
                "./apps/ai_interview",
                "./apps/ai_interview/smart_turn/coreml",
                "./apps/ai_interview/api",
            ]
            
            for search_path in search_paths:
                if os.path.exists(search_path):
                    for temp_file in temp_files:
                        file_path = Path(search_path) / temp_file
                        if file_path.exists() and file_path.is_file():
                            try:
                                file_path.unlink()
                                deleted_count += 1
                                print(f"[FILE] Deleted temporary WAV file: {file_path}")
                            except Exception as e:
                                print(f"[FILE ERROR] Failed to delete {file_path}: {e}")
            
            # Also clean up any WAV files with user_id in filename
            for search_path in search_paths:
                if os.path.exists(search_path):
                    try:
                        path_obj = Path(search_path)
                        for wav_file in path_obj.glob(f"*{user_id}*.wav"):
                            if wav_file.is_file():
                                wav_file.unlink()
                                deleted_count += 1
                                print(f"[FILE] Deleted user-specific WAV file: {wav_file}")
                    except Exception as e:
                        print(f"[FILE ERROR] Error searching for user WAV files in {search_path}: {e}")
                        
        except Exception as e:
            print(f"[FILE CLEANUP ERROR] {user_id}: {e}")
            
        return deleted_count

# Global instance
cleanup_manager = DisconnectCleanupManager()