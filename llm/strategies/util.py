import os
import json
import logging
import numpy as np
import redis


import time
import re
import hashlib
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Query Classification (Optional)
# -------------------------------------------------------------------
async def classify_query(query):
    """
    Dummy classification. Adjust as needed.
    """
    try:
        if not any([
            query.get("description"),
            query.get("role"),
            query.get("skills"),
            query.get("min_experience"),
            query.get("max_experience")
        ]):
            logger.warning("Empty or invalid query!")
            return "Invalid Query"
        elif (query.get("role") and query.get("skills")
              and query.get("max_experience") and query.get("min_experience")
              and not query.get("description")):
            return "Simple Query"
        elif (query.get("description")
              and (not query.get("role") or not query.get("skills") or not query.get("location"))):
            return "Complex Query"
        else:
            return "Straightforward Query"
    except Exception as e:
        logger.error(f"Error classifying query: {e}", exc_info=True)
        return "Invalid Query"

class QueryClassificationError:
    pass

def route_retrieval(state):
    strategy = state.get("strategy")
    if strategy == "Straightforward Query":
        return "structured_retrieval"
    elif strategy == "Simple Query":
        return "simple_query_retrieval"
    elif strategy == "Complex Query":
        return "complex_query_retrieval"
    else:
        raise QueryClassificationError(f"Unknown strategy: {strategy}")


    
    
    