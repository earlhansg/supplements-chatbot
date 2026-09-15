"""
Test wiring: point the app at a throwaway Redis, then seed it once per session.

**The import order in this file is load-bearing and is the whole reason it
looks like this.** `app/config.py` builds its `settings` singleton at import
time and `app/redis_client.py` opens its connection at import time from
`settings.redis_url`. By the time a fixture could monkeypatch either, the
connection is already open against the wrong server. So the redirect happens at
module level here, before the first `import app.*` — pytest imports a
directory's `conftest.py` before any test module in it, which is what makes
this early enough. `monkeypatch.setenv("REDIS_URL", ...)` inside a fixture is
too late and would silently test the developer's demo instance.

Real environment variables outrank `.env` values in pydantic-settings v2, so
this wins without touching anyone's `.env`.

Per-test *setting* overrides are a different story and `monkeypatch.setattr(
settings, "field", value)` is correct for them: `app/guardrails.py` and
`app/semantic_cache.py` read `settings.<field>` lazily at call time. Only
import-time-derived state — the Redis client — resists patching.
"""

import os

TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://127.0.0.1:6380")
os.environ["REDIS_URL"] = TEST_REDIS_URL

import pytest  # noqa: E402

# --- safety rail, before any app import ---------------------------------------
# This suite creates, overwrites and DELETES kb:*, cache:*, guardrail:*,
# cache:stats:* and rl:* keys. Port 6379 is the demo instance in
# docker-compose.yml, whose keyspace someone is probably mid-demo with.
# ALLOW_DEV_REDIS is the deliberate escape hatch, for an instance on :6379 you
# are genuinely willing to lose — never the demo one.
if ":6379" in TEST_REDIS_URL and not os.environ.get("ALLOW_DEV_REDIS"):
    pytest.exit(
        f"Refusing to run against {TEST_REDIS_URL}: this suite deletes kb:*, cache:*, "
        "guardrail:*, cache:stats:* and rl:* keys, and :6379 is the demo instance.\n"
        "Start the throwaway one instead:\n"
        "    docker compose --profile test up -d redis-test\n"
        "It listens on :6380, which is the default. Set ALLOW_DEV_REDIS=1 only if you "
        "genuinely mean the instance on :6379.",
        returncode=4,
    )

from app.config import settings  # noqa: E402
from app.guardrails import create_guardrail_index, load_guardrail_examples  # noqa: E402
from app.knowledge_base import create_kb_index, load_faqs  # noqa: E402
from app.redis_client import redis_client  # noqa: E402
from app.semantic_cache import create_cache_index, save_cache  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FAQS_PATH = os.path.join(DATA_DIR, "faqs.json")
GUARDRAIL_EXAMPLES_PATH = os.path.join(DATA_DIR, "guardrail_examples.json")


def redis_available() -> bool:
    try:
        return bool(redis_client.ping())
    except Exception:
        return False


# A module-level marker rather than a fixture, so a Redis-less run *skips*
# cleanly at collection instead of erroring inside every test body.
requires_redis = pytest.mark.skipif(
    not redis_available(),
    reason=f"no Redis Stack at {TEST_REDIS_URL} — docker compose --profile test up -d redis-test",
)


# Seeding embeds 10 FAQs + 27 exemplars, so the first Redis-touching test pays
# the model load (~10-20 s cold). Session scope keeps that to once per run.
@pytest.fixture(scope="session")
def seeded_kb() -> int:
    """`idx:kb` with the 10 demo FAQs in it. Anchors need this."""
    create_kb_index()
    return load_faqs(FAQS_PATH)


@pytest.fixture(scope="session")
def seeded_guardrails() -> int:
    """`idx:guardrail` with the 27 labelled exemplars in it."""
    create_guardrail_index()
    return load_guardrail_examples(GUARDRAIL_EXAMPLES_PATH)


@pytest.fixture(scope="session")
def cache_index() -> None:
    """`idx:cache`, created but never seeded — every test writes its own entries."""
    create_cache_index()


def _delete_matching(pattern: str) -> None:
    keys = list(redis_client.scan_iter(match=pattern, count=500))
    if keys:
        redis_client.delete(*keys)


@pytest.fixture(autouse=True)
def clean_cache():
    """Wipe cache entries, stats counters and rate-limit windows between tests.

    `cache:stats:*` goes too, unlike the documented cache-clearing command for
    the demo instance: there the counters are a measurement worth preserving,
    here a leaked counter makes the next test's arithmetic assertions fail for
    reasons that have nothing to do with the test. `rl:*` goes for the same
    reason — a fixed window that survives leaves the next test pre-limited.
    """
    if redis_available():
        _delete_matching("cache:*")
        _delete_matching("rl:*")
    yield


def write_cache_entry(
    query: str,
    answer: str,
    kb_anchor: str | None = None,
    kb_anchor_margin: float | None = None,
) -> str:
    """`save_cache()` plus the key it wrote, which save_cache does not return.

    Embeds symmetrically, exactly as the workflow's `embed_question` node does —
    the cache compares question to question, so both sides must skip the bge
    query prefix.
    """
    from app.embeddings import generate_embedding

    before = set(redis_client.scan_iter(match="cache:*", count=500))
    save_cache(query, answer, generate_embedding(query), kb_anchor, kb_anchor_margin)
    written = set(redis_client.scan_iter(match="cache:*", count=500)) - before

    # save_cache mints a uuid internally, so the key is recovered by difference.
    # Exactly one new key, or the helper is lying to its caller about which
    # entry the assertions below are about.
    assert len(written) == 1, f"expected exactly one new cache key, got {written}"
    return written.pop()


__all__ = ["requires_redis", "settings", "redis_client", "write_cache_entry"]
