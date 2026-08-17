"""
LangGraph workflow:

    START -> check_cache --(hit)--> END
                 |
              (miss)
                 v
          retrieve_context -> generate_answer -> save_cache -> END
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.knowledge_base import retrieve_context
from app.llm import generate_answer
from app.semantic_cache import check_cache, save_cache


class ChatState(TypedDict, total=False):
    question: str
    answer: str
    is_cached: bool
    cache_similarity: float
    context: list[dict]


def check_cache_node(state: ChatState) -> dict:
    hit = check_cache(state["question"])

    if hit:
        print(f'CACHE HIT (similarity={hit["similarity"]:.3f}) for: "{state["question"]}"')
        return {"answer": hit["answer"], "is_cached": True, "cache_similarity": hit["similarity"]}

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


def save_cache_node(state: ChatState) -> dict:
    save_cache(state["question"], state["answer"])
    return {}


def build_chat_workflow():
    graph = StateGraph(ChatState)

    graph.add_node("check_cache", check_cache_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("save_cache", save_cache_node)

    graph.add_edge(START, "check_cache")
    graph.add_conditional_edges(
        "check_cache", route_after_cache_check, {"hit": END, "miss": "retrieve_context"}
    )
    graph.add_edge("retrieve_context", "generate_answer")
    graph.add_edge("generate_answer", "save_cache")
    graph.add_edge("save_cache", END)

    return graph.compile()


chat_workflow = build_chat_workflow()
