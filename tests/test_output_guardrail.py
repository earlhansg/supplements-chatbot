"""
`should_cache()` — all four refusal reasons, their precedence, and the pass case.

Pure: no Redis, no KB, no LLM. `should_cache` gates the cache *write* only, so
every assertion here is about what gets enshrined for CACHE_TTL_SECONDS, never
about what the user sees — a rejected answer still reaches them.
"""

import pytest

from app.config import settings
from app.guardrails import REFUSAL_PATTERNS, should_cache


def grounded_context() -> list[dict]:
    """Context whose best match clears the grounding floor."""
    return [
        {"id": "faq-006", "similarity": settings.kb_grounding_floor + 0.3},
        {"id": "faq-005", "similarity": settings.kb_grounding_floor + 0.1},
    ]


def long_answer(prefix: str = "") -> str:
    """An answer comfortably past `min_cacheable_answer_chars`.

    Built from the setting rather than typed to a fixed length, so a tightened
    bound does not silently turn these into `too_short` assertions — which is
    precisely the trap the precedence test at the bottom pins down.
    """
    return prefix + "x" * (settings.min_cacheable_answer_chars + 20)


# --- generation_failed ---------------------------------------------------------


@pytest.mark.parametrize("answer", [None, "", "   ", "\n\t "], ids=["none", "empty", "spaces", "ws"])
def test_empty_generations_are_not_cached(answer):
    """`None` is genuinely reachable: the OpenAI SDK types `message.content` as `str | None`."""
    assert should_cache(answer, grounded_context()) == (False, "generation_failed")


def test_none_answer_never_reaches_len():
    """Ordering guard: a `None` that fell through to the length check would raise."""
    ok, reason = should_cache(None, [])
    assert (ok, reason) == (False, "generation_failed")


# --- too_short -----------------------------------------------------------------


def test_short_answer_is_not_cached():
    short = "x" * (settings.min_cacheable_answer_chars - 1)
    assert should_cache(short, grounded_context()) == (False, "too_short")


def test_length_is_measured_after_stripping():
    padded = "  " + "x" * (settings.min_cacheable_answer_chars - 1) + "  "
    assert should_cache(padded, grounded_context()) == (False, "too_short")


def test_answer_at_the_length_boundary_is_cacheable():
    """`<` not `<=`: exactly `min_cacheable_answer_chars` is long enough."""
    exact = "x" * settings.min_cacheable_answer_chars
    assert should_cache(exact, grounded_context()) == (True, None)


# --- refusal -------------------------------------------------------------------


@pytest.mark.parametrize("pattern", REFUSAL_PATTERNS)
def test_every_refusal_pattern_is_caught(pattern):
    """Parametrised over the constant, so a pattern added later gets coverage for free."""
    answer = long_answer(prefix=f"Sorry, {pattern} about that. ")
    assert should_cache(answer, grounded_context()) == (False, "refusal")


@pytest.mark.parametrize("pattern", REFUSAL_PATTERNS)
def test_refusal_matching_is_case_insensitive(pattern):
    answer = long_answer(prefix=f"Sorry, {pattern.upper()} about that. ")
    assert should_cache(answer, grounded_context()) == (False, "refusal")


# --- ungrounded ----------------------------------------------------------------


def test_empty_context_is_ungrounded():
    """Nothing was retrieved, so nothing grounded the answer — never grounded by default."""
    assert should_cache(long_answer(), []) == (False, "ungrounded")


def test_context_below_the_grounding_floor_is_ungrounded():
    weak = [{"id": "faq-001", "similarity": settings.kb_grounding_floor - 0.01}]
    assert should_cache(long_answer(), weak) == (False, "ungrounded")


def test_grounding_uses_the_best_match_not_the_first():
    """`max()`, not `[0]` — one strong match is enough to ground an answer."""
    mixed = [
        {"id": "faq-001", "similarity": settings.kb_grounding_floor - 0.2},
        {"id": "faq-002", "similarity": settings.kb_grounding_floor + 0.2},
    ]
    assert should_cache(long_answer(), mixed) == (True, None)


def test_grounding_floor_boundary_is_inclusive():
    at_the_line = [{"id": "faq-001", "similarity": settings.kb_grounding_floor}]
    assert should_cache(long_answer(), at_the_line) == (True, None)


# --- the pass case and rule precedence -----------------------------------------


def test_a_long_grounded_answer_is_cacheable():
    assert should_cache(long_answer("Standard shipping takes 3-5 business days. "), grounded_context()) == (
        True,
        None,
    )


def test_short_refusal_reports_too_short_not_refusal():
    """Rule precedence, which is the trap in this function.

    A short refusal trips both checks; `too_short` runs first and wins. Written
    down because the natural way to build a refusal fixture is a short string,
    and a test that asserted `refusal` against one would be passing by accident.
    """
    short_refusal = "I don't know"
    assert len(short_refusal) < settings.min_cacheable_answer_chars
    assert should_cache(short_refusal, grounded_context()) == (False, "too_short")


def test_ungrounded_refusal_reports_refusal_not_ungrounded():
    """`refusal` is checked before grounding, so it wins over an empty context."""
    assert should_cache(long_answer("I don't know. "), []) == (False, "refusal")
