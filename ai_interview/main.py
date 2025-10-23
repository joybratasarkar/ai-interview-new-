# app.py (your main FastAPI entry point)

from fastapi import FastAPI
import redis.asyncio as aioredis
import asyncio
import json
from fastapi.middleware.cors import CORSMiddleware
from ai_interview.api import websocket, routes
from ai_interview.services.shared import manager  
from ai_interview.config import REDIS_URL
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(websocket.router)
# app.include_router(websocket_natural.router)
# app.include_router(routes.router)

# Redis channel
CHANNEL_NAME = "interview_updates"
redis_listener_task = None
pubsub = None

@app.on_event("startup")
async def start_redis_subscriber():
    global redis_listener_task,pubsub

    redis = aioredis.from_url(REDIS_URL)
    pubsub = redis.pubsub()

    await pubsub.subscribe(CHANNEL_NAME)

    print(f"🟢 Subscribed to Redis channel: '{CHANNEL_NAME}'")

    async def reader():
        print(f"🟢 Starting Redis listener on channel '{CHANNEL_NAME}'")
        while True:
            try:
                # pubsub = redis.pubsub()

                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg and msg.get("data"):
                    raw = msg["data"]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")

                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        print(f"❌ Received invalid JSON: {raw}")
                        continue

                    # print(f"📥 Received Redis message: {payload}")

                    room_id = payload.get("room_id")
                    if room_id:
                        # Check if user is disconnected - don't send messages
                        if redis.get(f"disconnected:{room_id}"):
                            print(f"⚠️ User {room_id} is disconnected, skipping message delivery")
                            continue
                            
                        await manager.send_to_room(room_id, json.dumps(payload))
                        # if sent:
                        #     print(f"📤 Sent to WebSocket room: {room_id}")
                        # else:
                        #     print(f"⚠️ No active WebSocket in room '{room_id}'")
                    else:
                        print("⚠️ Missing 'room_id' in payload. Ignoring.")
            except Exception as e:
                print(f"❌ Redis message error: {e}")
            await asyncio.sleep(0.01)

    # Start background listener
    redis_listener_task = asyncio.create_task(reader())