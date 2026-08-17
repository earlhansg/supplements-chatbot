import redis

from app.config import settings

# Single shared connection, decode_responses=True so JSON/TEXT/TAG fields come
# back as plain Python str. Vector bytes only ever travel as raw query PARAMS,
# never as a decoded response field, so this is safe for FT.SEARCH KNN too.
redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
