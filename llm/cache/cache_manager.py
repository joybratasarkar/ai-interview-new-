import redis
import hashlib
import json

class CacheManager:
    def __init__(self, 
                 host='redis-18544.c8.us-east-1-2.ec2.redns.redis-cloud.com', 
                 port=18544, 
                 username="default", 
                 password="NPGoYkcfAOBRxJnLaLsWHuVkISo3yGcd"):
        self.client = redis.Redis(
            host=host,
            port=port,
            username=username,
            password=password,
            decode_responses=True
        )

    def generate_cache_key(self, data: dict) -> str:
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def get_cache(self, key: str):
        return self.client.get(key)

    def set_cache(self, key: str, value, ttl=3600):
        self.client.set(key, json.dumps(value), ex=ttl)

# # Example usage:
# if __name__ == "__main__":
#     cache = CacheManager()
#     data = {"user": "Alice", "action": "login"}
#     key = cache.generate_cache_key(data)
#     
#     # Set a value with a TTL of 1 hour (3600 seconds)
#     cache.set_cache(key, {"status": "success", "message": "User logged in."})
#     
#     # Retrieve and print the cached value
#     cached_value = cache.get_cache(key)
#     print("Cached value:", cached_value)
