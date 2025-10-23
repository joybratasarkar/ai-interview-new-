import os
from celery import Celery
from ai_interview.config import REDIS_URL
from dotenv import load_dotenv
import asyncio
import redis.asyncio as aioredis

load_dotenv()

CELERY_BROKER_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "ai_interviewer",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.task_routes = {
    "ai_interview.tasks.audio.process_audio": {"queue": "audio_queue"},
    "ai_interview.tasks.video.process_video": {"queue": "av_queue"},
    "ai_interview.tasks.natural_interview_flow.process_natural_interview": {"queue": "audio_queue"},

}

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
)


# ✅ Clean Redis on Celery reload (worker startup)
# @celery_app.on_after_configure.connect
# def cleanup_on_start(sender, **kwargs):
#     async def clear_all_audio_states():
#         print("[INFO] Cleaning Redis keys on Celery startup...")
#         try:
#             redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
#             keys_to_delete = []
#             for pattern in ["interview:ended:*", "lock:*", "stream:*"]:
#                 keys_to_delete += await redis_client.keys(pattern)

#             if keys_to_delete:
#                 await redis_client.delete(*keys_to_delete)
#                 print(f"[INFO] Deleted {len(keys_to_delete)} keys from Redis")
#             await redis_client.close()
#         except Exception as e:
#             print(f"[ERROR] Redis cleanup failed: {e}")

    # try:
        # loop = asyncio.get_event_loop()
        # loop.run_until_complete(clear_all_audio_states())
    # except RuntimeError:
    #     asyncio.run(clear_all_audio_states())




# ✅ Register task modules
import ai_interview.tasks.audio
import ai_interview.tasks.video
# import ai_interview.tasks.conversation_flow
# import ai_interview.tasks.natural_conversation_flow
# import ai_interview.tasks.langgraph_natural_flow
import ai_interview.tasks.natural_interview_flow

celery_app.autodiscover_tasks(['ai_interview.tasks'])
