"""Per-user Token Bucket rate limiter (assignment requirement)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from backend.app.config import get_settings
from backend.app.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Bucket:
    tokens: float
    last_refill: float


class TokenBucketRateLimiter:
    """In-memory token bucket. Swap for Redis in multi-instance deploys."""

    def __init__(self, capacity: float | None = None, refill_per_sec: float | None = None):
        settings = get_settings()
        self.capacity = capacity if capacity is not None else settings.rate_limit_tokens
        self.refill_per_sec = (
            refill_per_sec if refill_per_sec is not None else settings.rate_limit_refill_per_sec
        )
        self._buckets: dict[str, Bucket] = {}
        self._lock = asyncio.Lock()

    def _refill(self, bucket: Bucket, now: float) -> None:
        elapsed = now - bucket.last_refill
        bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_sec)
        bucket.last_refill = now

    async def allow(self, user_id: str, cost: float = 1.0) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        async with self._lock:
            now = time.monotonic()
            bucket = self._buckets.get(user_id)
            if bucket is None:
                bucket = Bucket(tokens=self.capacity, last_refill=now)
                self._buckets[user_id] = bucket
            self._refill(bucket, now)
            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True, 0.0
            deficit = cost - bucket.tokens
            retry_after = deficit / self.refill_per_sec if self.refill_per_sec > 0 else 60.0
            logger.warning("rate_limited", user_id=user_id, retry_after=retry_after)
            return False, retry_after


rate_limiter = TokenBucketRateLimiter()
