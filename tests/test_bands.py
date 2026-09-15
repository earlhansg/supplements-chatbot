"""
`classify_match()` — every branch, with no Redis and no KB.

These are the fast tests. They construct their inputs *relative to* the current
settings rather than hardcoding 0.78 / 0.90, because those are tunables: a test
that pins a tunable's value fails whenever someone tunes it, which is a failure
for the wrong reason and trains people to ignore the suite.

One import-time caveat, stated rather than hidden: importing
`app.semantic_cache` pulls in `app.knowledge_base` and therefore
`app.embeddings`, which constructs the SentenceTransformer at module level. So
this file needs no Redis and no network, but it does load the (locally cached)
embedding model. Breaking that chain would mean restructuring app code these
tests exist to protect.
"""

import pytest

from app.config import settings
from app.semantic_cache import classify_match


class RecordingAnchor:
    """A `resolve_incoming` that remembers whether it was called.

    The confident path's "exactly one KNN" property was a comment in
    `check_cache` for two phases. Injecting the lookup is what turns it into
    something a test can assert, and this class is the assertion.
    """

    def __init__(self, result: dict | None = None):
        self.result = result
        self.calls = 0

    def __call__(self) -> dict | None:
        self.calls += 1
        return self.result


def anchor(faq_id: str, margin: float) -> dict:
    """The shape `knowledge_base.anchor_for()` returns."""
    return {"id": faq_id, "similarity": 0.9, "margin": margin}


# --- the lower bound -----------------------------------------------------------


def test_below_lower_threshold_is_a_miss():
    resolve = RecordingAnchor(anchor("faq-001", 0.5))
    decision = classify_match(
        settings.cache_similarity_threshold - 0.01, "faq-001", 0.5, resolve
    )

    assert decision["band"] == "miss"
    # Nothing below the threshold is worth an anchor lookup either.
    assert resolve.calls == 0


def test_lower_threshold_is_inclusive():
    """`>=`, matching the runtime — exactly at the threshold is a candidate."""
    decision = classify_match(
        settings.cache_similarity_threshold,
        "faq-001",
        0.5,
        RecordingAnchor(anchor("faq-001", 0.5)),
    )

    assert decision["band"] != "miss"


# --- band 1: confident ---------------------------------------------------------


def test_confident_band_issues_no_anchor_lookup():
    """THE single-KNN contract. Not an optimisation — the fast path's whole cost model."""
    resolve = RecordingAnchor(anchor("faq-002", 0.5))
    decision = classify_match(
        settings.cache_hit_threshold_high, "faq-001", 0.5, resolve
    )

    assert decision["band"] == "confident"
    assert resolve.calls == 0, "the confident path must return before any KB lookup"


def test_confident_band_reports_no_anchors():
    """No comparison ran, so neither anchor id is reported — same rule as `_verdict()`."""
    decision = classify_match(
        settings.cache_hit_threshold_high + 0.05,
        "faq-001",
        0.5,
        RecordingAnchor(anchor("faq-002", 0.5)),
    )

    assert decision["entry_anchor"] is None
    assert decision["incoming_anchor"] is None


# --- the kill switch -----------------------------------------------------------


def test_kill_switch_serves_the_grey_band_unverified(monkeypatch):
    monkeypatch.setattr(settings, "cache_verify_grey_band", False)
    resolve = RecordingAnchor(anchor("faq-002", 0.5))

    decision = classify_match(grey(), "faq-001", 0.5, resolve)

    assert decision["band"] == "unverified"
    assert resolve.calls == 0, "the kill switch must skip the lookup, not just its result"


# --- the self-healing rule -----------------------------------------------------


@pytest.mark.parametrize(
    "entry_anchor,entry_margin",
    [
        (None, 0.5),      # pre-Phase-3 document: no anchor recorded
        ("faq-001", None),  # anchor recorded, margin not
        (None, None),
    ],
    ids=["no-anchor", "no-margin", "neither"],
)
def test_entry_without_anchor_fields_is_a_miss(entry_anchor, entry_margin):
    """A legacy entry has nothing to verify against, so it is re-answered.

    A miss rather than an unchecked hit: re-answering overwrites the gap with a
    document that carries both fields, so the cache heals one entry at a time
    instead of staying permanently unverifiable.
    """
    resolve = RecordingAnchor(anchor("faq-001", 0.5))

    decision = classify_match(grey(), entry_anchor, entry_margin, resolve)

    assert decision["band"] == "miss"
    assert resolve.calls == 0, "an unverifiable entry needs no incoming anchor"


# --- the empty-KB rule ---------------------------------------------------------


def test_empty_kb_serves_unverified():
    """Reachable mid-reseed, and not a reason to reject a match that cleared the bar."""
    resolve = RecordingAnchor(None)

    decision = classify_match(grey(), "faq-001", 0.5, resolve)

    assert decision["band"] == "unverified"
    assert resolve.calls == 1


# --- the margin guard ----------------------------------------------------------


def test_undecided_incoming_anchor_may_not_veto():
    """The Phase 3 finding, and the assertion most likely to break on a "simplification".

    The anchors DISAGREE here. A coin-flip anchor still may not reject the
    match, because a number that was never meaningful must not outvote a
    similarity that already cleared the threshold.
    """
    undecided = settings.cache_anchor_margin_min - 0.01
    decision = classify_match(
        grey(), "faq-001", 0.5, RecordingAnchor(anchor("faq-006", undecided))
    )

    assert decision["band"] == "unverified"


def test_undecided_entry_anchor_may_not_veto():
    """The write side of the same guard — the reason both margins are stored."""
    undecided = settings.cache_anchor_margin_min - 0.01
    decision = classify_match(
        grey(), "faq-001", undecided, RecordingAnchor(anchor("faq-006", 0.5))
    )

    assert decision["band"] == "unverified"


def test_a_zero_margin_is_a_real_value_not_unknown():
    """0.0 is a dead tie — decisively undecided. `None` is "never recorded".

    They take different paths (unverified vs miss), and conflating them is how
    a legacy document would start being served unchecked.
    """
    assert (
        classify_match(grey(), "faq-001", 0.0, RecordingAnchor(anchor("faq-006", 0.5)))["band"]
        == "unverified"
    )
    assert (
        classify_match(grey(), "faq-001", None, RecordingAnchor(anchor("faq-006", 0.5)))["band"]
        == "miss"
    )


def test_margin_guard_boundary_is_inclusive():
    """Exactly at `cache_anchor_margin_min` the anchor counts as decisive (`<`, not `<=`)."""
    at_the_line = settings.cache_anchor_margin_min
    decision = classify_match(
        grey(), "faq-001", at_the_line, RecordingAnchor(anchor("faq-006", at_the_line))
    )

    assert decision["band"] == "rejected"


# --- the anchor comparison -----------------------------------------------------


def test_decisive_agreeing_anchors_verify():
    decisive = settings.cache_anchor_margin_min + 0.10
    decision = classify_match(
        grey(), "faq-007", decisive, RecordingAnchor(anchor("faq-007", decisive))
    )

    assert decision["band"] == "verified"
    assert decision["entry_anchor"] == "faq-007"
    assert decision["incoming_anchor"] == "faq-007"


def test_decisive_disagreeing_anchors_reject():
    """The 0.809 shipping false hit, reduced to its decision."""
    decisive = settings.cache_anchor_margin_min + 0.10
    decision = classify_match(
        grey(), "faq-006", decisive, RecordingAnchor(anchor("faq-005", decisive))
    )

    assert decision["band"] == "rejected"
    assert decision["entry_anchor"] == "faq-006"
    assert decision["incoming_anchor"] == "faq-005"


def test_the_incoming_anchor_is_resolved_at_most_once():
    """Called once on the grey path, never twice — it is a Redis round trip plus an encode."""
    decisive = settings.cache_anchor_margin_min + 0.10
    resolve = RecordingAnchor(anchor("faq-007", decisive))

    classify_match(grey(), "faq-007", decisive, resolve)

    assert resolve.calls == 1


# --- collapsing the band entirely ----------------------------------------------


def test_the_grey_band_can_be_collapsed_to_nothing(monkeypatch):
    """With the high threshold dropped onto the low one, every hit is confident."""
    monkeypatch.setattr(
        settings, "cache_hit_threshold_high", settings.cache_similarity_threshold
    )
    resolve = RecordingAnchor(anchor("faq-005", 0.5))

    decision = classify_match(
        settings.cache_similarity_threshold, "faq-006", 0.5, resolve
    )

    assert decision["band"] == "confident"
    assert resolve.calls == 0


def grey() -> float:
    """A similarity squarely inside the grey band, wherever the two bounds are."""
    return (settings.cache_similarity_threshold + settings.cache_hit_threshold_high) / 2
