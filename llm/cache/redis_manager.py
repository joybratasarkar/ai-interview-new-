"""
redis_manager.py

A generic, reusable Redis manager to:
  - Create dynamic indexes
  - Store data with optional embedding
  - Retrieve data via text + vector queries
"""

import json
import logging
import hashlib
import numpy as np

from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from llm.factory import get_redis_client

logger = logging.getLogger(__name__)

class RedisManager:
    def __init__(self):
        self.redis_client = get_redis_client()
        print("Testing Redis connection...")
        print('redis_client',self.redis_client)
        print(self.redis_client.ping())  # ✅ Should return True
    def initialize_index(self, index_name: str, schema_fields: list, prefix="jd:", drop_if_exists=False):
        definition = IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)

        if drop_if_exists:
            try:
                self.redis_client.ft(index_name).dropindex(delete_documents=False)
                logger.info(f"Dropped existing Redis index '{index_name}'.")
            except Exception as e:
                logger.warning(f"Could not drop index '{index_name}': {e}")

        # Check or create
        try:
            self.redis_client.ft(index_name).info()
            logger.info(f"Index '{index_name}' already exists; skipping creation.")
        except Exception:
            logger.info(f"Creating Redis index '{index_name}'.")
            self.redis_client.ft(index_name).create_index(schema_fields, definition=definition)
            logger.info(f"✅ Redis index '{index_name}' created.")

    def store_data(
        self,
        index_name: str,
        prefix: str,
        data: dict,
        embed_model=None,
        vector_field=None,
        embed_field="description",
        expire_sec=3600
    ):
        combined_text = json.dumps(data, ensure_ascii=False)
        hashed = hashlib.md5(combined_text.encode("utf-8")).hexdigest()
        redis_key = f"{prefix}{hashed}"

        # Optional embedding
        vector_bytes = b""
        if embed_model and vector_field:
            text_val = data.get(embed_field, "")
            vec = embed_model.encode(text_val)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vector_bytes = vec.astype(np.float32).tobytes()

        # Convert fields
        mapping = {"combined_text": combined_text.encode("utf-8")}
        for k, v in data.items():
            if isinstance(v, list):
                joined = ", ".join(str(x) for x in v)
                mapping[k] = joined.lower()
            elif isinstance(v, (str, int, float)):
                mapping[k] = str(v).lower() if isinstance(v, str) else v
            else:
                mapping[k] = str(v).lower()

        if vector_field and embed_model:
            mapping[vector_field] = vector_bytes

        self.redis_client.hset(redis_key, mapping=mapping)
        self.redis_client.expire(redis_key, expire_sec)
        logger.info(f"✅ Stored data in Redis key={redis_key} under index '{index_name}'.")

    def compute_similarity(self, embed_model, text1: str, text2: str) -> float:
        v1 = embed_model.encode(text1)
        v2 = embed_model.encode(text2)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 0:
            v1 /= n1
        if n2 > 0:
            v2 /= n2
        return float(np.dot(v1, v2))

    def get_vector(self, doc_id, field_name):
        """
        Retrieve a vector field from a Redis document.
    
        Args:
            doc_id (str): The document ID.
            field_name (str): The vector field name.
    
        Returns:
            bytes or None: The stored vector bytes if found, else None.
        """
        try:
            return self.redis_client.hget(doc_id, field_name)
        except Exception as e:
            logger.error(f"❌ Error retrieving vector field '{field_name}' for doc '{doc_id}': {e}")
            return None

    def retrieve_data(
        self,
        index_name: str,
        text_query: str = "*",
        vector_query: dict = None,
        top_k: int = 1,
        key:str='combined_text'
    ):
        """
        Retrieve data from Redis using both text filtering and vector similarity search.

        Args:
            index_name (str): The name of the Redis index.
            text_query (str): The text query for filtering results.
            vector_query (dict): Dictionary containing vector search details.
            top_k (int): Number of results to retrieve.

        Returns:
            list[dict]: Retrieved job descriptions from Redis.
        """

        query_str = text_query.strip() if text_query.strip() else "*"
        
        params_dict = {}
        print("vector_query------------------------------------joy",vector_query)
        print('index_name',index_name)
        if vector_query and "field" in vector_query and "vector" in vector_query:
            field = vector_query["field"]
            vec_val = vector_query["vector"]
            k = vector_query.get("K", top_k)

            norm = np.linalg.norm(vec_val)
            if norm > 0:
                vec_val /= norm
            vec_bytes = vec_val.astype(np.float32).tobytes()

            query_str = f"({query_str}) =>[KNN {k} @{field} $vec_param]"
            params_dict["vec_param"] = vec_bytes
        print("query_str--joy",query_str)
        q = (
            Query(query_str)
            .return_fields("id", key)  
            .paging(0, top_k)
            .with_scores()
            .dialect(2)
        )

        try:
            print("params_dict",params_dict)
            print("index_name",index_name)
            results = self.redis_client.ft(index_name).search(q, query_params=params_dict)
            print("results",results)
            if results.total == 0:
                logger.info(f"🔍 No docs found for query='{query_str}'.")
                return []

            docs = []
            for doc in results.docs:
                fields = vars(doc)
                doc_data = {
                    "id": doc.id,  # Fix: Store ID in result
                    "combined_text": fields.get("combined_text", "")
                }
                docs.append(doc_data)

            logger.info(f"✅ Found {len(docs)} doc(s) with query='{query_str}' in index='{index_name}'.")
            return docs
        except Exception as e:
            logger.error(f"❌ Redis search error in index='{index_name}': {e}")
            return []
