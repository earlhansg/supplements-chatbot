"""
The counters, the per-entry hit total, and the TTL property the design rests on.

The autouse `clean_cache` fixture in conftest deletes `cache:stats:*` before
each test here. Without that, lifetime counters leak between tests and the
arithmetic assertions in `read_stats` fail for reasons that look like a bug in
`read_stats`.
"""

import pytest

from app import metrics
from app.config import settings
from app.redis_client import redis_client
from app.semantic_cache import CACHE_PREFIX

from conftest import requires_redis, write_cache_entry

pytestmark = [requires_redis, pytest.mark.redis, pytest.mark.embeddings]

ANSWER = "Standard shipping typically takes 3-5 business days."


@pytest.fixture(autouse=True)
def _cache_index(cache_index):
    """`read_stats` counts documents via FT.SEARCH, so the index must exist."""


def counter(name: str) -> int:
    return int(redis_client.get(f"{metrics.STATS_PREFIX}{name}") or 0)


# --- record_hit ----------------------------------------------------------------


def test_record_hit_increments_lifetime_daily_and_per_entry():
    key = write_cache_entry("How long does shipping take?", ANSWER, "faq-006", 0.2)

    assert metrics.record_hit(key) == 1

    assert counter("hits") == 1
    assert counter(f"{metrics._today()}:hits") == 1
    assert redis_client.json().get(key, "$.hits") == [1]


def test_record_hit_returns_the_running_per_entry_total():
    key = write_cache_entry("How long does shipping take?", ANSWER, "faq-006", 0.2)

    assert [metrics.record_hit(key) for _ in range(3)] == [1, 2, 3]
    assert counter("hits") == 3


def test_numincrby_does_not_reset_the_ttl():
    """The property the whole per-entry-counter design rests on.

    If `JSON.NUMINCRBY` refreshed the key's TTL, a popular entry would become
    immortal and the 24 h expiry would quietly stop meaning anything. Measured
    rather than assumed, and compared read-to-read rather than by sleeping —
    the two reads are what matters, not the wall clock between them.
    """
    key = write_cache_entry("How long does shipping take?", ANSWER, "faq-006", 0.2)

    before = redis_client.ttl(key)
    for _ in range(5):
        metrics.record_hit(key)
    after = redis_client.ttl(key)

    assert after <= before, "JSON.NUMINCRBY must not extend the entry's TTL"


def test_record_hit_on_a_pre_metrics_document_returns_none():
    """`JSON.NUMINCRBY` with a `$`-path replies with an empty list when nothing matched.

    A document written before cache entries carried `hits` therefore degrades
    to "no count" instead of raising — and the lifetime counter still moves,
    because the request was still a hit.
    """
    key = f"{CACHE_PREFIX}pre-metrics-entry"
    redis_client.json().set(key, "$", {"query": "q", "answer": ANSWER})

    assert metrics.record_hit(key) is None
    assert counter("hits") == 1


# --- record_miss / record_blocked ----------------------------------------------


def test_record_miss_increments_lifetime_and_daily():
    metrics.record_miss()

    assert counter("misses") == 1
    assert counter(f"{metrics._today()}:misses") == 1


def test_record_blocked_has_no_daily_twin():
    """Deliberate: GET /stats surfaces only hits and misses for "today".

    A counter nothing reads is the scope creep this repo refuses.
    """
    metrics.record_blocked()

    assert counter("blocked") == 1
    assert not redis_client.exists(f"{metrics.STATS_PREFIX}{metrics._today()}:blocked")


# --- the daily key's TTL -------------------------------------------------------


def test_the_daily_key_expires():
    metrics.record_miss()

    ttl = redis_client.ttl(f"{metrics.STATS_PREFIX}{metrics._today()}:misses")
    assert settings.stats_daily_ttl_seconds - 5 <= ttl <= settings.stats_daily_ttl_seconds


def test_a_second_increment_does_not_extend_the_daily_ttl():
    """EXPIRE runs only on the first increment of the day.

    Re-expiring per request would slide the day forward and the key would never
    age out — the same rule app/rate_limit.py applies to its window.
    """
    daily = f"{metrics.STATS_PREFIX}{metrics._today()}:misses"

    metrics.record_miss()
    before = redis_client.ttl(daily)
    for _ in range(5):
        metrics.record_miss()
    after = redis_client.ttl(daily)

    assert after <= before


def test_the_lifetime_counters_carry_no_ttl():
    """They are lifetime totals; only the per-day twins expire."""
    metrics.record_miss()

    assert redis_client.ttl(f"{metrics.STATS_PREFIX}misses") == -1


# --- read_stats ----------------------------------------------------------------


def test_read_stats_arithmetic_excludes_blocked_from_the_denominator():
    for _ in range(3):
        metrics.record_miss()
    metrics.record_blocked()
    key = write_cache_entry("How long does shipping take?", ANSWER, "faq-006", 0.2)
    metrics.record_hit(key)

    stats = metrics.read_stats()

    assert (stats["hits"], stats["misses"], stats["blocked"]) == (1, 3, 1)
    # Blocked questions never reached the cache, so the hit rate answers "of the
    # questions the cache was asked, how many did it answer?".
    assert stats["total"] == 4
    assert stats["hit_rate"] == pytest.approx(0.25)
    assert stats["llm_calls_avoided"] == stats["hits"]
    assert stats["today"] == {"hits": 1, "misses": 3}


def test_hit_rate_is_zero_rather_than_a_zero_division():
    stats = metrics.read_stats()

    assert stats["total"] == 0
    assert stats["hit_rate"] == 0.0


def test_cache_entries_matches_the_index():
    write_cache_entry("How long does shipping take?", ANSWER, "faq-006", 0.2)
    write_cache_entry("What is your refund policy?", ANSWER, "faq-001", 0.2)

    # The same FT.SEARCH view the dashboard's own panel gets, so /stats and the
    # cache panel can never disagree.
    assert metrics.read_stats()["cache_entries"] == 2


def test_the_stats_counters_are_never_indexed():
    """They live under `idx:cache`'s `cache:` prefix but are plain strings.

    A JSON index ignores non-JSON keys, so they never appear in a KNN result —
    the one consequence being that a `--scan --pattern 'cache:*'` sweep matches
    them, which is why the documented clear command filters them out.
    """
    metrics.record_miss()
    write_cache_entry("How long does shipping take?", ANSWER, "faq-006", 0.2)

    assert metrics.read_stats()["cache_entries"] == 1
