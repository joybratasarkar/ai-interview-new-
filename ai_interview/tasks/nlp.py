from ai_interview.celery_app import celery_app
from ai_interview.config import REDIS_URL
import redis
import json


publisher = redis.Redis.from_url(REDIS_URL)

@celery_app.task
def process_answer(payload: dict) -> None:
    """
    payload = { "room_id": "<interview_42>", "text": "Candidate answer text" }
    """
    room_id = payload["room_id"]
    answer_text = payload["text"]

    # … do heavy NLP here (e.g. sentiment analysis) …
    sentiment = "positive" if "good" in answer_text.lower() else "neutral"

    message = {
        "room_id":   room_id,
        "task_type": "answer",
        "result":    {"sentiment": sentiment, "length": len(answer_text)},
    }
    publisher.publish(REDIS_URL, json.dumps(message))