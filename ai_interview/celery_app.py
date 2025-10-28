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
    "ai_interview.tasks.natural_interview_flow.process_natural_interview": {"queue": "interview_queue"},

}

# ✅ Register task modules
import ai_interview.tasks.audio
import ai_interview.tasks.video
# import ai_interview.tasks.conversation_flow
# import ai_interview.tasks.natural_conversation_flow
# import ai_interview.tasks.langgraph_natural_flow
import ai_interview.tasks.natural_interview_flow


celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
)



celery_app.autodiscover_tasks(['ai_interview.tasks'])
