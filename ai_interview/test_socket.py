import websockets
import asyncio
import json

async def interactive_ws():
    uri = "ws://localhost:8009/ws/user123"
    async with websockets.connect(uri) as websocket:
        print("✅ Connected to WebSocket server.")

        async def receive_messages():
            while True:
                message = await websocket.recv()
                try:
                    data = json.loads(message)
                    print(f"\n📥 Server → You: {data.get('result', {}).get('bot_response', message)}")
                except json.JSONDecodeError:
                    print(f"\n📥 Server → You (raw): {message}")

        async def send_messages():
            while True:
                user_input = input("\n💬 You: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("👋 Exiting WebSocket.")
                    break

                message = {
                    "type": "answer",
                    "text": user_input,
                    "room_id": "user123"
                }
                await websocket.send(json.dumps(message))

        await asyncio.gather(receive_messages(), send_messages())

asyncio.run(interactive_ws())
