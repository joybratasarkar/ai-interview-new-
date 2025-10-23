"""
redis_utils.py

Feature-specific wrapper around RedisManager for job descriptions.
- Initialize job index
- Store job
- Retrieve cached job
"""

import logging
import re
import numpy as np

from redis.commands.search.field import TextField, NumericField, VectorField
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
import json
from llm.cache.redis_manager import RedisManager
from llm.factory import get_embedding_model
from llm.exceptions import RetrievalError

logger = logging.getLogger(__name__)

# We'll define a global 'manager' instance for job postings
manager = RedisManager()

# Name of the index for jobs
JOB_INDEX_NAME = "llm_jd_index"
JOB_PREFIX = "jd:"
EMBED_DIM = 384  # matches your model dimension

def initialize_redis_index(drop_if_exists=False):
    """
    Initialize the job postings index with a custom schema for job data.
    """
    schema_fields = [
        TextField("role"),
        TextField("skills"),
        TextField("tone"),
        TextField("language"),
        TextField("industry"),
        TextField("location"),
        TextField("job_type"),
        TextField("employment_type"),
        NumericField("min_experience"),
        NumericField("max_experience"),
        VectorField("description_vector", "FLAT", {
            "TYPE": "FLOAT32",
            "DIM": EMBED_DIM,
            "DISTANCE_METRIC": "COSINE"
        })
    ]
    manager.initialize_index(JOB_INDEX_NAME, schema_fields, prefix=JOB_PREFIX, drop_if_exists=drop_if_exists)

def store_jd_in_redis(jd_data: dict, user_query: dict, embed_model, expire_sec=3600):
    """
    Store a job description doc + user query, merging them if needed,
    then storing in Redis with a vector embedding on 'description'.
    """
    # Combine them if needed
    merged_data = dict(jd_data)
    merged_data.update(user_query)

    # Optional fallback for 'description'
    desc = merged_data.get("description", "")
    if not desc:
        logger.warning("⚠️ No 'description' in data. Setting default.")
        merged_data["description"] = "No description provided."

    manager.store_data(
        index_name=JOB_INDEX_NAME,
        prefix=JOB_PREFIX,
        data=merged_data,
        embed_model=embed_model,
        vector_field="description_vector",  # matches the schema
        embed_field="description",
        expire_sec=expire_sec
    )


def retrieve_cached_jd(query_fields: dict, embed_model, base_similarity_threshold=1.0):
    """
    Retrieve a cached job description from Redis that matches the user's query fields,
    using both text filtering and vector similarity search.

    Args:
        query_fields (dict): The user's original query fields.
        embed_model: The embedding model for generating the query vector.
        base_similarity_threshold (float): Threshold for cosine similarity validation.

    Returns:
        dict or None: The stored JD if found and above threshold, else None.
    """

    # 1️⃣ **Normalize Fields**
    # print("query_fields---------------------------------------------------------------------------------",query_fields)
    role = query_fields.get("role")
    role = role.strip().lower() if isinstance(role, str) else None
    # print("role---------------------------------------------------------------------------------",role)
    location = query_fields.get("location")
    location = location.strip().lower() if isinstance(location, str) else None
    # print("location---------------------------------------------------------------------------------",location)
    job_type = query_fields.get("job_type")
    job_type = job_type.strip().lower() if isinstance(job_type, str) else None
    # print("job_type---------------------------------------------------------------------------------",job_type)
    language = query_fields.get("language")
    language = language.strip().lower() if isinstance(language, str) else None
    # print("language---------------------------------------------------------------------------------",language)
    employment_type = query_fields.get("employment_type")
    employment_type = employment_type.strip().lower() if isinstance(employment_type, str) else None
    # print("employment_type---------------------------------------------------------------------------------",employment_type)
    tone = query_fields.get("tone")
    tone = tone.strip().lower() if isinstance(tone, str) else None
    # print("tone---------------------------------------------------------------------------------",tone)
    industry = query_fields.get("industry")
    industry = industry.strip().lower() if isinstance(industry, str) else None

    # Ensure experience is always an integer (default 0)
    try:
        min_exp = int(query_fields.get("min_experience", 0) or 0)
    except ValueError:
        min_exp = 0  # Default if conversion fails

    try:
        max_exp = int(query_fields.get("max_experience", 0) or 0)
    except ValueError:
        max_exp = 0  # Default if conversion fails

    # Handling skills (Ensure it's a list and convert to lowercase)
    skills_val = query_fields.get("skills", [])
    if isinstance(skills_val, list):
        skills_str = ", ".join([skill.strip().lower() for skill in skills_val if isinstance(skill, str)]) or None
    elif isinstance(skills_val, str):
        skills_str = skills_val.strip().lower() or None
    else:
        skills_str = None  # Default if skills are invalid


    # 2️⃣ **Compute vector embedding for description**
    # print("query_fields---------------------------------------------------------------------------------",query_fields)
    description = query_fields.get("description", "")
    # print("description---------------------------------------------------------------------------------",description)
    if not description:
        description = "No description provided."
        logger.info("❌ 'description' field is missing or empty in the query.")

    query_vec = embed_model.encode(description)
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec /= norm
    else:
        logger.warning("Zero norm encountered for query description embedding.")

    # 3️⃣ **Build Redis Query (Text + KNN Search)**
    # text_query = (
    #     f'(@role:"{role}" '
    #     f'@location:"{location}" '
    #     f'@job_type:"{job_type}" '
    #     f'@language:"{language}" '
    #     f'@tone:"{tone}" '
    #     f'@industry:"{industry}" '
    #     f'@skills:"{skills_str}" '
    #     f'@employment_type:"{employment_type}" '
    #     f'@min_experience:[0 {min_exp}] '
    #     f'@max_experience:[{max_exp} +inf])'
    # )
    query_fields = {
    "role": f'@role:"{role}"' if role else None,
    "location": f'@location:"{location}"' if location else None,
    "job_type": f'@job_type:"{job_type}"' if job_type else None,
    "language": f'@language:"{language}"' if language else None,
    "tone": f'@tone:"{tone}"' if tone else None,
    "industry": f'@industry:"{industry}"' if industry else None,
    "skills_str": f'@skills:"{skills_str}"' if skills_str else None,
    "employment_type": f'@employment_type:"{employment_type}"' if employment_type else None,
    "min_experience": f'@min_experience:[0 {min_exp}]' if min_exp is not None else None,
    "max_experience": f'@max_experience:[{max_exp} +inf]' if max_exp is not None else None
}
    print("joy---------------------------------------------------------------------------------",query_fields)
    # Filter out None values and join query parts dynamically
    text_query = "(" + " ".join(filter(None, query_fields.values())) + ")"
    vector_query = {"field": "description_vector", "vector": query_vec, "K": 1}
    print("text_query---------------------------------------------------------------------------------",text_query)
    # 4️⃣ **Retrieve Data from Redis**
    docs = manager.retrieve_data(index_name=JOB_INDEX_NAME, text_query=text_query, vector_query=vector_query, top_k=1)

    if not docs:
        logger.info("🔍 No cached JD found in Redis for this query.")
        return None

    # 5️⃣ **Ensure `doc_id` is correctly retrieved**
    doc = docs[0]
    doc_id = doc.get("id", None)  # Fix: Retrieve doc_id correctly
    if not doc_id:
        logger.error("❌ Retrieved document is missing an ID.")
        return None

    logger.debug(f"🔍 Retrieved document ID: {doc_id}")

    # 6️⃣ **Retrieve the Stored Vector Separately**
    cached_vector_bytes = manager.get_vector(doc_id, "description_vector")
    if not cached_vector_bytes:
        logger.error(f"❌ Stored vector missing in Redis for doc_id: {doc_id}")
        return None

    # 7️⃣ **Validate & Process Data**
    return parse_and_validate_stored_data(
        doc_fields=doc,
        cached_vector_bytes=cached_vector_bytes,
        query_vec=query_vec,
        base_similarity_threshold=base_similarity_threshold,
        min_exp=min_exp,
        max_exp=max_exp,
        location=location
    )


def parse_and_validate_stored_data(doc_fields, cached_vector_bytes, query_vec, base_similarity_threshold, min_exp, max_exp, location):
    """
    Parses stored data, retrieves vector, validates cosine similarity, and updates necessary fields.

    Args:
        doc_fields (dict): Retrieved document fields from Redis.
        cached_vector_bytes (bytes): Stored vector bytes retrieved from RedisManager.
        query_vec (numpy array): Query vector for similarity check.
        base_similarity_threshold (float): Threshold for cosine similarity validation.
        min_exp (int): Minimum experience required.
        max_exp (int): Maximum experience required.
        location (str): Location filter.

    Returns:
        dict or None: Processed job description if similarity is above threshold, else None.
    """
    stored_data = {}

    # 1️⃣ **Decode `combined_text` from Redis**
    combined_text_val = doc_fields.get("combined_text")
    if not combined_text_val:
        logger.error("❌ `combined_text` missing in Redis document.")
        return None

    try:
        if isinstance(combined_text_val, bytes):
            combined_text_val = combined_text_val.decode("utf-8")
        stored_data = json.loads(combined_text_val)
    except Exception as e:
        logger.error(f"❌ Error decoding combined_text: {e}")
        return None

    # 2️⃣ **Convert Stored Vector to Writable NumPy Array**
    if not cached_vector_bytes:
        logger.error("❌ Stored vector missing in Redis document.")
        return None

    try:
        cached_vector = np.frombuffer(cached_vector_bytes, dtype=np.float32).copy()  # ✅ Fix: Make it writable
        cached_norm = np.linalg.norm(cached_vector)
        if cached_norm > 0:
            cached_vector /= cached_norm
    except Exception as e:
        logger.error(f"❌ Error processing stored vector: {e}")
        return None

    # 3️⃣ **Compute Cosine Similarity**
    try:
        sim_score = sk_cosine_similarity([query_vec], [cached_vector])[0][0]
        logger.info(f"🔍 Cosine similarity: {sim_score:.4f}")
    except Exception as e:
        logger.error(f"❌ Error computing cosine similarity: {e}")
        return None

    if round(sim_score * 100) < round(base_similarity_threshold * 100):
        logger.info(f"❌ Similarity below threshold ({base_similarity_threshold}). Not returning cached JD.")
        return None

    # 4️⃣ **Remove Unnecessary Fields**
    fields_to_remove = ["role", "min_experience", "max_experience", "skills", "description", "language", "tone", "industry"]
    for field in fields_to_remove:
        stored_data.pop(field, None)

    # 5️⃣ **Update Experience & Location**
    needs_update = (
        stored_data.get("min_experience") != min_exp
        or stored_data.get("max_experience") != max_exp
        or stored_data.get("location") != location
    )

    if needs_update:
        stored_data["min_experience"] = min_exp
        stored_data["max_experience"] = max_exp
        stored_data["experience"] = f"{min_exp}-{max_exp} years"
        stored_data["location"] = location

    # 6️⃣ **Update Summary & Qualifications**
    experience_pattern = r"(\d+)-(\d+)"

    if "summary" in stored_data and isinstance(stored_data["summary"], str):
        stored_data["summary"] = re.sub(experience_pattern, stored_data["experience"], stored_data["summary"])

    if "qualifications" in stored_data and isinstance(stored_data["qualifications"], list):
        for i, qualification in enumerate(stored_data["qualifications"]):
            if isinstance(qualification, str):
                stored_data["qualifications"][i] = re.sub(experience_pattern, stored_data["experience"], qualification)

    logger.info("✅ Cosine similarity is above threshold. Returning cached JD.")
    return stored_data
