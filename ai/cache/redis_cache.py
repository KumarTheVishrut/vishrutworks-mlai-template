import redis.asyncio as redis
from config import settings

r = redis.from_url(settings.REDIS_URL)

async def get_cached(key: str) -> str | None:
    value = await r.get(key)
    return value.decode() if value else None

async def set_cached(key: str, value: str, ttl: int = 300):
    await r.setex(key, ttl, value.encode())
