"""
`save_cache()` + `check_cache()` against a real Redis Stack instance.

Similarity bounds here are deliberately LOOSE. Pinning 0.809 to three decimals
would turn a torch or sentence-transformers upgrade into what looks like a
logic regression; the band and the hit/miss decision are the contract, and the
score is evidence for it.

Nothing asserts that an expired entry has left `FT.SEARCH`: on Redis 7.4
expiration is not accounted for when computing a result set, so such an
assertion is flaky by construction. TTL and EXISTS are the honest checks.
"""

import pytest

from app.config import settings
from app.embeddings import generate_embedding
from app.redis_client import redis_client
from app.semantic_cache import CACHE_PREFIX, check_cache

from conftest import requires_redis, write_cache_entry

pytestmark = [requires_redis, pytest.mark.redis, pytest.mark.embeddings]

SHIPPING_TIME = "How long does shipping take?"
SHIPPING_COST = "How much does shipping cost?"
ANSWER = "Standard shipping typically takes 3-5 business days within the continental US."

# Comfortably past `cache_anchor_margin_min`, so the anchors in these tests are
# decisive and the margin guard is not the thing under test here.
DECISIVE = 0.20


@pytest.fixture(autouse=True)
def _redis_state(seeded_kb, cache_index):
    """Anchors need a seeded idx:kb; check_cache needs idx:cache to exist."""


def ask(question: str) -> dict:
    """What the workflow does: embed once, symmetrically, then look up."""
    return check_cache(question, generate_embedding(question))


# --- the write -----------------------------------------------------------------


def test_save_cache_writes_every_field_and_a_ttl():
    key = write_cache_entry(SHIPPING_TIME, ANSWER, "faq-006", DECISIVE)

    document = redis_client.json().get(key)
    assert set(document) == {
        "query", "answer", "embedding",
        "created_at", "hits", "kb_anchor", "kb_anchor_margin", "model",
    }
    assert document["query"] == SHIPPING_TIME
    assert document["answer"] == ANSWER
    assert len(document["embedding"]) == settings.embedding_dim
    assert document["hits"] == 0
    assert document["kb_anchor"] == "faq-006"
    assert document["model"] == settings.embedding_model

    ttl = redis_client.ttl(key)
    # A window rather than equality: EXPIRE and the read are not atomic.
    assert settings.cache_ttl_seconds - 5 <= ttl <= settings.cache_ttl_seconds


# --- the bands -----------------------------------------------------------------


def test_an_exact_repeat_is_a_confident_hit():
    key = write_cache_entry(SHIPPING_TIME, ANSWER, "faq-006", DECISIVE)

    verdict = ask(SHIPPING_TIME)

    assert verdict["hit"] is True
    assert verdict["band"] == "confident"
    assert verdict["similarity"] > 0.99
    assert verdict["key"] == key
    assert verdict["answer"] == ANSWER


def test_an_unrelated_question_is_a_plain_miss():
    write_cache_entry(SHIPPING_TIME, ANSWER, "faq-006", DECISIVE)

    verdict = ask("Are your supplements third-party tested for quality and safety?")

    assert verdict["hit"] is False
    # `None`, not "rejected": nothing was close enough to have a band decided.
    assert verdict["band"] is None
    assert verdict["similarity"] is None


def test_an_empty_cache_does_not_raise():
    """The first-ever question, and the `not results.docs` branch."""
    verdict = ask(SHIPPING_TIME)

    assert verdict["hit"] is False
    assert verdict["band"] is None


def test_the_grey_band_rejects_the_documented_false_hit():
    """THE assertion this suite exists for.

    With the shipping *time* question cached, asking about shipping *cost*
    scores ~0.809 — inside the grey band, and served the wrong answer before
    Phase 3. The two anchors (faq-006 vs faq-005) disagree decisively, so it is
    now declined, and the near-miss score survives the return so the caller can
    report a rejection rather than silence.
    """
    write_cache_entry(SHIPPING_TIME, ANSWER, "faq-006", DECISIVE)

    verdict = ask(SHIPPING_COST)

    assert verdict["hit"] is False
    assert verdict["band"] == "rejected"
    assert 0.79 < verdict["similarity"] < 0.83
    assert verdict["entry_anchor"] == "faq-006"
    assert verdict["incoming_anchor"] == "faq-005"


def test_the_kill_switch_turns_that_rejection_into_a_hit(monkeypatch):
    """`cache_verify_grey_band=False` is the pre-verification behaviour, exactly.

    The band is still reported, which is why the response shape never varies
    with the flag and the two modes stay comparable against one client.
    """
    monkeypatch.setattr(settings, "cache_verify_grey_band", False)
    write_cache_entry(SHIPPING_TIME, ANSWER, "faq-006", DECISIVE)

    verdict = ask(SHIPPING_COST)

    assert verdict["hit"] is True
    assert verdict["band"] == "unverified"
    assert verdict["answer"] == ANSWER


def test_an_undecided_anchor_serves_the_grey_band_unverified():
    """A coin-flip write-side margin may not veto, even against a disagreeing anchor."""
    undecided = settings.cache_anchor_margin_min - 0.01
    write_cache_entry(SHIPPING_TIME, ANSWER, "faq-006", undecided)

    verdict = ask(SHIPPING_COST)

    assert verdict["hit"] is True
    assert verdict["band"] == "unverified"


# --- legacy documents ----------------------------------------------------------


def legacy_entry(query: str, answer: str) -> str:
    """A pre-Phase-3 cache document: no `kb_anchor`, no `kb_anchor_margin`.

    Written through `json().set` directly rather than `save_cache`, because
    `save_cache` cannot produce this shape any more — which is the point. A
    24 h TTL means mixed-shape documents are the NORMAL state for a day after
    any deploy, not a hypothetical.
    """
    key = f"{CACHE_PREFIX}legacy-test-entry"
    redis_client.json().set(
        key,
        "$",
        {"query": query, "answer": answer, "embedding": generate_embedding(query)},
    )
    redis_client.expire(key, settings.cache_ttl_seconds)
    return key


def test_a_legacy_entry_is_a_miss_in_the_grey_band():
    """Self-healing: re-answering overwrites the gap with a verifiable document."""
    legacy_entry(SHIPPING_TIME, ANSWER)

    verdict = ask(SHIPPING_COST)

    assert verdict["hit"] is False
    assert verdict["band"] is None


def test_a_legacy_entry_still_hits_in_the_confident_band():
    """Above `cache_hit_threshold_high` nothing is verified, so age is irrelevant."""
    key = legacy_entry(SHIPPING_TIME, ANSWER)

    verdict = ask(SHIPPING_TIME)

    assert verdict["hit"] is True
    assert verdict["band"] == "confident"
    assert verdict["key"] == key


# --- an empty knowledge base ---------------------------------------------------


def test_an_empty_kb_serves_the_grey_band_unverified(monkeypatch):
    """Reachable mid-reseed. `anchor_for` returns None and the match is served anyway."""
    monkeypatch.setattr("app.semantic_cache.anchor_for", lambda question: None)
    write_cache_entry(SHIPPING_TIME, ANSWER, "faq-006", DECISIVE)

    verdict = ask(SHIPPING_COST)

    assert verdict["hit"] is True
    assert verdict["band"] == "unverified"
