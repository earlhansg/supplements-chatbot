"""
All three graph paths end to end, with a stubbed LLM.

**Patch `app.workflow.generate_answer`, never `app.llm_local.generate_answer`.**
`app/workflow.py` binds the name into its own module namespace at import, so
patching the source module leaves the compiled graph calling the real backend —
a trap whose symptom is a test that hangs for the local server's 120 s timeout
instead of failing. No test in this file may require a reachable LLM; that is
verified by running the suite with the LLM server stopped.

`chat_workflow` is compiled at import, so tests invoke that object directly
rather than rebuilding the graph.
"""

import pytest

from app import metrics, workflow
from app.redis_client import redis_client

from conftest import requires_redis

pytestmark = [requires_redis, pytest.mark.redis, pytest.mark.embeddings]

GENERATED = (
    "Standard shipping typically takes 3-5 business days within the continental US, "
    "and expedited shipping takes 1-2 business days."
)


@pytest.fixture(autouse=True)
def _redis_state(seeded_kb, seeded_guardrails, cache_index):
    """The graph touches all three collections, so all three must be seeded."""


@pytest.fixture
def llm(monkeypatch):
    """A stub standing in for the LLM, counting how often the graph called it."""

    class Stub:
        def __init__(self):
            self.calls = []
            self.answer = GENERATED

        def __call__(self, question: str, context: list[dict]) -> str:
            self.calls.append(question)
            return self.answer

    stub = Stub()
    monkeypatch.setattr(workflow, "generate_answer", stub)
    return stub


def counter(name: str) -> int:
    return int(redis_client.get(f"{metrics.STATS_PREFIX}{name}") or 0)


def cache_size() -> int:
    return len(list(redis_client.scan_iter(match="cache:*", count=500))) - stats_keys()


def stats_keys() -> int:
    """`cache:stats:*` shares the `cache:` prefix, so it has to be discounted."""
    return len(list(redis_client.scan_iter(match="cache:stats:*", count=500)))


# --- the blocked path ----------------------------------------------------------


def test_a_blocked_question_costs_no_llm_call_and_never_searches_the_cache(llm):
    before = cache_size()

    result = workflow.chat_workflow.invoke(
        {"question": "Is creatine safe to take alongside my lisinopril?"}
    )

    assert result["guardrail"]["action"] == "block"
    assert result["is_cached"] is False
    # The canned exemplar response, not a generation.
    assert result["answer"] == result["guardrail"]["response"]
    assert llm.calls == []
    assert counter("blocked") == 1
    assert (counter("hits"), counter("misses")) == (0, 0)
    assert cache_size() == before, "a blocked question must never write to idx:cache"


# --- the miss path -------------------------------------------------------------


def test_a_novel_question_generates_caches_and_counts_one_miss(llm):
    result = workflow.chat_workflow.invoke({"question": "How long does shipping take?"})

    assert result["is_cached"] is False
    assert result["answer"] == GENERATED
    assert result["cached_now"] is True
    assert len(llm.calls) == 1
    assert cache_size() == 1
    assert (counter("misses"), counter("hits")) == (1, 0)


def test_the_cached_entry_records_the_anchor_the_miss_path_already_retrieved(llm):
    workflow.chat_workflow.invoke({"question": "How long does shipping take?"})

    key = next(
        k for k in redis_client.scan_iter(match="cache:*") if not k.startswith("cache:stats:")
    )
    document = redis_client.json().get(key)

    # Free: the miss path's KNN 3 over idx:kb produced both numbers already.
    assert document["kb_anchor"] == "faq-006"
    assert document["kb_anchor_margin"] is not None


# --- the hit path --------------------------------------------------------------


def test_repeating_a_question_hits_without_calling_the_llm_again(llm):
    workflow.chat_workflow.invoke({"question": "How long does shipping take?"})
    result = workflow.chat_workflow.invoke({"question": "How long does shipping take?"})

    assert result["is_cached"] is True
    assert result["cache_band"] == "confident"
    assert result["answer"] == GENERATED
    assert len(llm.calls) == 1, "the second request must not reach the LLM"
    assert result["cached_hits"] == 1
    assert (counter("hits"), counter("misses")) == (1, 1)
    assert cache_size() == 1, "a hit writes no new entry"


# --- the rejection path --------------------------------------------------------


def test_a_grey_band_rejection_counts_as_exactly_one_miss(llm):
    """The invariant the terminal-node design exists to guarantee.

    A request that *begins* as a cache match and is then declined must count
    once, as a miss — never as a hit, and never as both. It is also why the
    declined entry's own `$.hits` must not move: `record_hit_node` bumps
    whatever `cache_key` is in state, and `check_cache_node` deliberately sets
    none on a rejection.
    """
    workflow.chat_workflow.invoke({"question": "How long does shipping take?"})
    declined_key = next(
        k for k in redis_client.scan_iter(match="cache:*") if not k.startswith("cache:stats:")
    )
    ttl_before = redis_client.ttl(declined_key)

    result = workflow.chat_workflow.invoke({"question": "How much does shipping cost?"})

    assert result["is_cached"] is False
    assert result["cache_band"] == "rejected"
    # The near-miss score survives the miss path — without it a rejection is
    # indistinguishable from an ordinary miss.
    assert 0.79 < result["rejected_similarity"] < 0.83
    assert (counter("hits"), counter("misses")) == (0, 2)
    assert redis_client.json().get(declined_key, "$.hits") == [0]
    assert redis_client.ttl(declined_key) <= ttl_before


# --- the output guardrail ------------------------------------------------------


def test_a_refusal_is_answered_but_not_cached_and_still_counts_one_miss(llm):
    """The second edge into `record_miss`: a suppressed write is still a miss."""
    llm.answer = "I don't know the answer to that, sorry — please contact support."

    result = workflow.chat_workflow.invoke({"question": "How long does shipping take?"})

    # Served to the user regardless: the output guardrail gates the WRITE only.
    assert result["answer"] == llm.answer
    assert result["not_cached_reason"] == "refusal"
    assert result.get("cached_now", False) is False
    assert cache_size() == 0
    assert counter("misses") == 1


def test_an_empty_generation_is_reported_as_generation_failed(llm):
    llm.answer = None

    result = workflow.chat_workflow.invoke({"question": "How long does shipping take?"})

    assert result["not_cached_reason"] == "generation_failed"
    assert cache_size() == 0
    assert counter("misses") == 1


# --- one counter per path ------------------------------------------------------


def test_every_path_increments_exactly_one_counter(llm):
    """Three requests, three outcomes, three counters — summing to three."""
    workflow.chat_workflow.invoke({"question": "How long does shipping take?"})  # miss
    workflow.chat_workflow.invoke({"question": "How long does shipping take?"})  # hit
    # Scores 0.897 against a medical exemplar — chosen for margin over the
    # threshold, not just for being blockable. An off-topic question like "What
    # is the capital of Peru?" sits at 0.435 and is one of the false negatives
    # scripts/eval_guardrail.py reports at the tuned threshold.
    workflow.chat_workflow.invoke(
        {"question": "Is creatine safe to take alongside my lisinopril?"}
    )  # blocked

    assert (counter("hits"), counter("misses"), counter("blocked")) == (1, 1, 1)
