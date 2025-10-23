from ai_interview.celery_app import celery_app
from ai_interview.config import REDIS_URL
import redis
import json




publisher = redis.Redis.from_url(REDIS_URL)

@celery_app.task(name="ai_interview.tasks.video.process_video")
def process_video(payload: dict) -> None:
    """
    payload = { "room_id": "<interview_42>", "blob": "<base64‐video‐blob>" }
    """
    room_id = payload["room_id"]
    video_blob = payload["blob"]

    # … do heavy video processing here (e.g. frame analysis) …
    result_summary = f"PROCESSED_VIDEO_LEN:{len(video_blob)}"

    message = {
        "room_id":   room_id,
        "task_type": "video",
        "result":    result_summary,
    }
    publisher.publish(REDIS_URL, json.dumps(message))
