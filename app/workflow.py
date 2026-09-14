"""
LangGraph workflow:

    START -> embed_question -> check_guardrail --(blocked)--> record_blocked -> END
                                    |
                                 (allowed)
                                    v
                               check_cache --(hit)--> record_hit -> END
                                    |
                                 (miss)
                                    v
                      retrieve_context -> generate_answer -> check_answer --(skip)--> record_miss -> END
                                                                 |                         ^
                                                              (cache)                      |
                                                                 v                         |
                                                            save_cache ---------------------

Every branch here is a real conditional edge, not an `if` inside a node — the
routing decisions are the thing the graph is showing off.

Each of the three outcomes ends in its own terminal recording node, so every
path through the graph increments exactly one counter (see `app/metrics.py`).
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.embeddings import generate_embedding
from app.guardrails import check_input, should_cache
from app.knowledge_base import retrieve_context
# from app.llm import generate_answer  # swap for app.llm_local to use the local server
from app.llm_local import generate_answer
from app.metrics import record_blocked, record_hit, record_miss
from app.semantic_cache import check_cache, save_cache


class ChatState(TypedDict, total=False):
    question: str
    # The one symmetric embedding of `question`, shared by the guardrail and
    # cache KNN searches. 768 floats — the only large field in state, so keep
    # everything else here small (scores and ids, not whole documents).
    question_embedding: list[float]
    answer: str
    is_cached: bool
    cache_similarity: float
    context: list[dict]
    guardrail: dict
    not_cached_reason: str
    cached_now: bool
    # The `cache:<uuid>` key of the entry a hit matched. Carried so the terminal
    # record_hit node can bump that entry's own counter without re-running the
    # search. An id, not a document — state stays small.
    cache_key: str
    # That entry's hit total after this request. None for an entry written
    # before cache documents carried a `hits` field.
    cached_hits: int


def embed_question_node(state: ChatState) -> dict:
    # Embedded once per request, symmetrically (no bge query prefix), because
    # both consumers compare question <-> question: the guardrail against its
    # exemplars and the cache against previously-asked questions. KB retrieval
    # deliberately re-embeds with is_query=True — that comparison is
    # question -> passage and needs the asymmetric prefix.
    return {"question_embedding": generate_embedding(state["question"])}


def check_guardrail_node(state: ChatState) -> dict:
    verdict = check_input(state["question"], state["question_embedding"])

    if verdict:
        print(f'GUARDRAIL BLOCK ({verdict["label"]}, similarity={verdict["similarity"]:.3f})')
        # is_cached is set here too so app/main.py can build a complete
        # response without the cache node ever having run.
        return {"answer": verdict["response"], "guardrail": verdict, "is_cached": False}

    return {}


def route_after_guardrail(state: ChatState) -> str:
    return "blocked" if state.get("guardrail") else "allowed"


def check_cache_node(state: ChatState) -> dict:
    hit = check_cache(state["question"], state["question_embedding"])

    if hit:
        print(f'CACHE HIT (similarity={hit["similarity"]:.3f}) for: "{state["question"]}"')
        return {
            "answer": hit["answer"],
            "is_cached": True,
            "cache_similarity": hit["similarity"],
            "cache_key": hit["key"],
        }

    print(f'CACHE MISS / NOT CACHED for: "{state["question"]}"')
    return {"is_cached": False}


def route_after_cache_check(state: ChatState) -> str:
    return "hit" if state["is_cached"] else "miss"


def retrieve_context_node(state: ChatState) -> dict:
    context = retrieve_context(state["question"])
    return {"context": context}


def generate_answer_node(state: ChatState) -> dict:
    answer = generate_answer(state["question"], state.get("context", []))
    return {"answer": answer}


def check_answer_node(state: ChatState) -> dict:
    """Output guardrail: decide whether this answer is fit to cache.

    A node rather than part of the routing function, because the reason string
    has to reach app/main.py through state — routing functions return a label
    and write nothing. It runs before the write and after the answer is
    settled, so a rejected answer still goes back to the user; only the cache
    entry is suppressed.
    """
    ok, reason = should_cache(state.get("answer"), state.get("context", []))

    if not ok:
        print(f"NOT CACHED (reason={reason})")
        return {"not_cached_reason": reason}

    return {}


def route_after_answer_check(state: ChatState) -> str:
    return "skip" if state.get("not_cached_reason") else "cache"


def save_cache_node(state: ChatState) -> dict:
    context = state.get("context", [])
    # context is sorted by ascending KNN distance, so [0] is the best FAQ match.
    # Free: the miss path already paid for this search.
    kb_anchor = context[0]["id"] if context else None
    save_cache(state["question"], state["answer"], state["question_embedding"], kb_anchor)
    return {"cached_now": True}


# Counting lives in terminal nodes, not inside check_cache, so that every path
# through the graph increments exactly one of hits / misses / blocked. It also
# keeps the door open for a path that *begins* as a cache hit and is then
# rejected — a rejection like that has to count as a single miss, which is
# unreachable once check_cache has already counted a hit.


def record_hit_node(state: ChatState) -> dict:
    return {"cached_hits": record_hit(state["cache_key"])}


def record_miss_node(state: ChatState) -> dict:
    record_miss()
    # Nothing to add to state — a node must still return a partial state dict,
    # and an empty one is how you say "I changed nothing".
    return {}


def record_blocked_node(state: ChatState) -> dict:
    record_blocked()
    return {}


def build_chat_workflow():
    graph = StateGraph(ChatState)

    graph.add_node("embed_question", embed_question_node)
    graph.add_node("check_guardrail", check_guardrail_node)
    graph.add_node("check_cache", check_cache_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("check_answer", check_answer_node)
    graph.add_node("save_cache", save_cache_node)
    graph.add_node("record_hit", record_hit_node)
    graph.add_node("record_miss", record_miss_node)
    graph.add_node("record_blocked", record_blocked_node)

    graph.add_edge(START, "embed_question")
    graph.add_edge("embed_question", "check_guardrail")
    # A blocked question ends here: no LLM call, and idx:cache is never searched.
    graph.add_conditional_edges(
        "check_guardrail",
        route_after_guardrail,
        {"blocked": "record_blocked", "allowed": "check_cache"},
    )
    graph.add_conditional_edges(
        "check_cache", route_after_cache_check, {"hit": "record_hit", "miss": "retrieve_context"}
    )
    graph.add_edge("retrieve_context", "generate_answer")
    graph.add_edge("generate_answer", "check_answer")
    graph.add_conditional_edges(
        "check_answer", route_after_answer_check, {"cache": "save_cache", "skip": "record_miss"}
    )
    # record_miss has two incoming edges — a written answer and a suppressed one
    # are both misses. Only one of the two paths runs per request, so it still
    # counts exactly once.
    graph.add_edge("save_cache", "record_miss")
    graph.add_edge("record_hit", END)
    graph.add_edge("record_miss", END)
    graph.add_edge("record_blocked", END)

    return graph.compile()


chat_workflow = build_chat_workflow()
