# apps/ai_interview/services/socket_manager.py

from typing import Dict, List
from fastapi import WebSocket
import redis.asyncio as aioredis
import asyncio

from apps.ai_interview.config import REDIS_URL

redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)

class ConnectionManager:
    def __init__(self):
        self.rooms: Dict[str, List[WebSocket]] = {}

    # async def connect(self, room_id: str, websocket: WebSocket):
    #     self.rooms.setdefault(room_id, []).append(websocket)
    #     print(f"🟢 WebSocket connected to room '{room_id}'. Total connections: {len(self.rooms[room_id])}")
    async def connect(self, room_id: str, websocket: WebSocket):
    # Close and remove any existing connection in the room
        if room_id in self.rooms and self.rooms[room_id]:
            for conn in self.rooms[room_id]:
                try:
                    await conn.close()
                except Exception as e:
                    print(f"⚠️ Error closing existing WebSocket for room '{room_id}': {e}")
            self.rooms[room_id].clear()
    
        # Add new connection
        self.rooms.setdefault(room_id, []).append(websocket)
        print(f"🟢 WebSocket connected to room '{room_id}'. Total connections: {len(self.rooms[room_id])}")


    async def disconnect(self, room_id: str, websocket: WebSocket):
        if room_id in self.rooms and websocket in self.rooms[room_id]:
            self.rooms[room_id].remove(websocket)
            print(f"🔴 WebSocket disconnected from room '{room_id}'. Remaining connections: {len(self.rooms.get(room_id, []))}")

            if not self.rooms.get(room_id):
                del self.rooms[room_id]
                print(f"🧹 Room '{room_id}' removed (no active connections).")

                # Delete any residual Redis stream key
                redis_key = f"stream:{room_id}"
                try:
                    await redis_client.delete(redis_key)
                    print(f"🧹 Redis stream/key '{redis_key}' deleted.")
                except Exception as e:
                    print(f"⚠️ Failed to delete Redis key '{redis_key}': {e}")

                # Mark interview as ended
                try:
                    await redis_client.set(f"interview:ended:{room_id}", "1")
                    await redis_client.expire(f"interview:ended:{room_id}", 3600)
                    print(f"🛑 Redis flag 'interview:ended:{room_id}' set.")
                except Exception as e:
                    print(f"⚠️ Failed to set interview-ended flag for '{room_id}': {e}")

    async def send_to_room(self, room_id: str, message: str) -> bool:
        if room_id not in self.rooms:
            return False

        sent = False
        dead_connections = []

        for connection in list(self.rooms[room_id]):
            # Skip closed or stale sockets
            if connection.client_state.name != "CONNECTED":
                print(f"⚠️ Skipping stale WebSocket (state={connection.client_state.name}) in room '{room_id}'")
                dead_connections.append(connection)
                continue

            try:
                await connection.send_text(message)
                sent = True
            except Exception as e:
                print(f"❌ Failed to send message to WebSocket in room '{room_id}': {e}")
                dead_connections.append(connection)

        # Clean up any dead connections
        for conn in dead_connections:
            await self.disconnect(room_id, conn)

        if not sent:
            print(f"⚠️ All WebSocket connections in room '{room_id}' were stale or closed.")

        return sent

    async def send_to_self(self, room_id: str, websocket: WebSocket, message: str):
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"❌ Failed to send self-message to '{room_id}': {e}")
            await self.disconnect(room_id, websocket)

    def list_active_rooms(self):
        print("📡 Active WebSocket Rooms:")
        for room, conns in self.rooms.items():
            print(f"  - {room}: {len(conns)} connection(s)")

    async def poll_and_send(self, task_id: str, room_id: str, task_type: str, timeout: int = 30):
        """Poll Celery task result and send to WebSocket when ready."""
        from apps.ai_interview.celery_app import celery_app
        import json
        
        print(f"[POLL] Starting to poll task {task_id} for room {room_id}")
        
        try:
            # Poll for task completion
            task_result = celery_app.AsyncResult(task_id)
            start_time = asyncio.get_event_loop().time()
            
            while not task_result.ready():
                if asyncio.get_event_loop().time() - start_time > timeout:
                    print(f"[POLL] Task {task_id} timed out after {timeout}s")
                    await self.send_to_room(room_id, json.dumps({
                        "error": f"Task {task_id} timed out after {timeout}s",
                        "task_type": task_type
                    }))
                    return
                await asyncio.sleep(0.1)  # Check every 100ms
            
            print(f"[POLL] Task {task_id} completed with status: {task_result.status}")
            
            # Task completed, send result
            if task_result.successful():
                result = task_result.result
                print(f"[POLL] Sending result to room {room_id}: {result}")
                response = {
                    "task_id": task_id,
                    "task_type": task_type,
                    "status": "completed",
                    "result": result
                }
                sent = await self.send_to_room(room_id, json.dumps(response))
                print(f"[POLL] Message sent successfully: {sent}")
            else:
                await self.send_to_room(room_id, json.dumps({
                    "task_id": task_id,
                    "task_type": task_type,
                    "status": "failed",
                    "error": str(task_result.result)
                }))
                
        except Exception as e:
            await self.send_to_room(room_id, json.dumps({
                "task_id": task_id,
                "task_type": task_type,
                "status": "error",
                "error": str(e)
            }))

# Single shared instance
manager = ConnectionManager()
