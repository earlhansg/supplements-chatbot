"""
Cache metrics — the counters that let the cache report on itself.

Three plain-string counters, incremented once per request in the graph's
terminal nodes:

    cache:stats:hits          INCR on every cache hit
    cache:stats:misses        INCR on every miss
    cache:stats:blocked       INCR on every guardrail refusal

plus a per-day pair that expires itself after STATS_DAILY_TTL_SECONDS:

    cache:stats:<YYYY-MM-DD>:hits
    cache:stats:<YYYY-MM-DD>:misses

and a counter living inside each cache document:

    JSON.NUMINCRBY cache:<uuid> $.hits 1

`JSON.NUMINCRBY` modifies the document in place and does **not** reset the
key's TTL, so counting a popular entry can never make it immortal. That
property is the reason the per-entry counter is safe to keep here rather than
in a parallel key that would have to be expired in lockstep.

The `cache:stats:` prefix sits under `idx:cache`'s `cache:` prefix on purpose —
the counters read as part of the cache feature in RedisInsight. They are plain
strings, and a JSON index ignores non-JSON keys, so they are never indexed and
never appear in a KNN result. (The one consequence: a `--scan --pattern
'cache:*'` sweep matches them too, which is why the documented cache-clearing
command filters them out.)

Dates are UTC. A server-local boundary would make "today" mean different things
to the API and to anyone reading the keys from another timezone.

This module is HTTP-agnostic like the other Redis modules — it raises nothing a
caller has to translate, and the graph's terminal nodes are its only writers.
"""

from datetime import datetime, timezone

from redis.commands.search.query import Query

from app.config import settings
from app.redis_client import redis_client
from app.semantic_cache import CACHE_INDEX

STATS_PREFIX = "cache:stats:"


def _today() -> str:
    """UTC date stamp for the per-day counter keys."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _bump(field: str) -> None:
    """INCR the global counter for `field` and its per-day twin."""
    redis_client.incr(f"{STATS_PREFIX}{field}")

    daily_key = f"{STATS_PREFIX}{_today()}:{field}"
    # Expire only on the first increment of the day, exactly as
    # app/rate_limit.py does for its window: re-expiring on every request would
    # slide the day forward and the key would never age out.
    if redis_client.incr(daily_key) == 1:
        redis_client.expire(daily_key, settings.stats_daily_ttl_seconds)


def record_hit(cache_key: str) -> int | None:
    """Count a cache hit and bump that entry's own hit counter.

    Returns the entry's new hit total, or None for a document written before
    this field existed. `JSON.NUMINCRBY` with a `$`-path replies with a list of
    new values — empty when the path matched nothing — so a pre-metrics entry
    degrades to "no count" instead of raising.
    """
    _bump("hits")
    result = redis_client.json().numincrby(cache_key, "$.hits", 1)
    return int(result[0]) if result else None


def record_miss() -> None:
    """Count a question that was answered from scratch rather than from cache."""
    _bump("misses")


def record_blocked() -> None:
    """Count a question the input guardrail refused."""
    # No per-day twin: GET /stats surfaces only hits and misses for "today",
    # and a counter with no reader is the scope creep this repo refuses.
    redis_client.incr(f"{STATS_PREFIX}blocked")


def _counter(*parts: str) -> int:
    """Read one counter. A key that was never incremented reads as 0, not None."""
    return int(redis_client.get(f"{STATS_PREFIX}{''.join(parts)}") or 0)


def read_stats() -> dict:
    """Everything GET /stats reports: lifetime counters, today's, and the live entry count."""
    hits = _counter("hits")
    misses = _counter("misses")
    blocked = _counter("blocked")

    # Blocked questions never reach the cache, so they are deliberately not in
    # the denominator: the hit rate answers "of the questions the cache was
    # asked, how many did it answer?", not "how many requests arrived?".
    total = hits + misses

    # FT.SEARCH rather than FT.INFO's num_docs: this is the same view the
    # dashboard's own FT.SEARCH gets, so /stats and the cache panel can never
    # disagree with each other.
    cache_entries = redis_client.ft(CACHE_INDEX).search(Query("*").paging(0, 0)).total

    today = _today()

    return {
        "hits": hits,
        "misses": misses,
        "blocked": blocked,
        "total": total,
        "hit_rate": hits / total if total else 0.0,
        # The same number as `hits`, under the name that states the point: every
        # hit is a generation that never happened.
        "llm_calls_avoided": hits,
        "cache_entries": cache_entries,
        "today": {"hits": _counter(today, ":hits"), "misses": _counter(today, ":misses")},
    }
