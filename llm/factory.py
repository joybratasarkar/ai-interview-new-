# llm/factory.py

"""
Factory methods to create or retrieve shared resources like
LLM clients, embedding models, and Redis clients.
"""

import os
from sentence_transformers import SentenceTransformer
from langchain_google_vertexai import ChatVertexAI
from redis import Redis
from dotenv import load_dotenv
load_dotenv()
# ----------------------------------------------------------------
# Global/Environment Config (adjust as needed or use .env files)
# ----------------------------------------------------------------
SERVICE_ACCOUNT_KEY = os.path.abspath("xooper.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY
PROJECT_ID = "xooper-450012"
LOCATION = "us-central1"

# For Redis:
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# Validate that Redis variables are correctly set
if not REDIS_HOST or not REDIS_PORT or not REDIS_PASSWORD:
    raise ValueError("❌ ERROR: Redis configuration missing! Check .env file.")

print(f"✅ Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
# ----------------------------------------------------------------
# LLM Factories
# ----------------------------------------------------------------

def get_vertex_ai_llm(
    model_name: str = "gemini-2.0-flash-lite-001",
    temperature: float = 0.7
):
    """
    Return an instance of Vertex AI Chat model.
    In the future, you can add logic for multiple LLM providers or
    environment-based config (e.g., dev vs. prod).
    """
    return ChatVertexAI(
        model=model_name,
        project=PROJECT_ID,
        location=LOCATION,
        temperature=temperature
    )


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-V2"
):
    """
    Return a SentenceTransformer embedding model.
    If you want to switch models, just change the name or add logic
    to pick a model based on environment variables.
    """
    return SentenceTransformer(model_name)


# ----------------------------------------------------------------
# Redis Factory
# ----------------------------------------------------------------

def get_redis_client(
    host: str = REDIS_HOST,
    port: int = REDIS_PORT,
    password: str = REDIS_PASSWORD,
    decode_responses: bool = False
):
    """
    Return a Redis client instance. 
    If you need special parameters (e.g., SSL, decode_responses=True),
    you can add them here or read from environment variables.
    """
    return Redis(host=host, port=port, password=password, decode_responses=decode_responses)
