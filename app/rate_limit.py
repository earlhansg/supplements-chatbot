"""
Fixed-window rate limiter, as a FastAPI dependency.

One Redis counter per (caller, window): `rl:<caller>:<window number>`, where
the window number is the current epoch second integer-divided by
`RATE_LIMIT_WINDOW_SECONDS`. `INCR` creates it, and an `EXPIRE` on the *first*
increment lets it clean itself up — which is also why the whole limiter needs
no background sweep.

**This limiter is effectively global, and that is honest rather than
accidental.** The browser never reaches FastAPI: every request arrives from
the Next.js proxy, so `request.client.host` is the proxy's address for all
users. The purpose here is bounding LLM spend — a cache miss is a
multi-second generation — not isolating callers from each other.

No forwarded-client-address header is read, deliberately. Trusting one would
be spoofable in an app with no auth *and* would imply per-user isolation this
topology cannot deliver. The honest version of per-caller limiting is a
trusted-proxy list plus a header the proxy itself sets, which is a larger
change than this limiter.

This module raises `HTTPException` even though the Redis modules stay
HTTP-agnostic: it *is* the HTTP layer, not a storage layer.
"""

import time

from fastapi import HTTPException, Request, status

from app.config import settings
from app.redis_client import redis_client

RATE_LIMIT_PREFIX = "rl:"


def enforce_rate_limit(request: Request) -> None:
    """Count this request against the caller's current window; raise 429 past the limit.

    Deliberately a plain `def` rather than a coroutine: it touches the
    synchronous Redis client, so FastAPI must run it in the worker threadpool
    where blocking is harmless, not directly on the event loop.
    """
    if not settings.rate_limit_enabled:
        return

    client = request.client.host if request.client else "unknown"
    window = int(time.time()) // settings.rate_limit_window_seconds
    key = f"{RATE_LIMIT_PREFIX}{client}:{window}"

    count = redis_client.incr(key)
    # Only on the first increment of a window. Re-expiring on every request
    # would slide the window forward and the limit would never reset.
    if count == 1:
        redis_client.expire(key, settings.rate_limit_window_seconds)

    if count > settings.rate_limit_max_requests:
        retry_after = max(redis_client.ttl(key), 1)
        raise HTTPException(
            # `detail` stays a plain string so the frontend's describeFailure()
            # unpacks it as-is (frontend/src/lib/backend.ts).
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: {settings.rate_limit_max_requests} requests per "
                f"{settings.rate_limit_window_seconds}s. Retry in {retry_after}s."
            ),
            headers={"Retry-After": str(retry_after)},
        )
