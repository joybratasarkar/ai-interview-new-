import redis

r = redis.Redis(
    host='dev.xooper.in',
    port=6379,
    password='Xooper#1234',
    db=0
)

r.flushdb()  # Clears only DB 0
# r.flushall()  # Use this to clear all databases
print("Redis DB 0 cleared.")
