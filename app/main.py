import json as jsonlib
from contextlib import asynccontextmanager
from pathlib import Path

import openai
from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.guardrails import GUARDRAIL_INDEX, create_guardrail_index, load_guardrail_examples
from app.knowledge_base import KB_INDEX, create_kb_index, load_faqs
from app.rate_limit import enforce_rate_limit
from app.schemas import ChatRequest, ChatResponse
from app.semantic_cache import create_cache_index
from app.workflow import chat_workflow

FAQS_PATH = Path(__file__).resolve().parent.parent / "data" / "faqs.json"
GUARDRAIL_EXAMPLES_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "guardrail_examples.json"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_kb_index()
    create_cache_index()
    create_guardrail_index()

    from app.redis_client import redis_client

    info = redis_client.ft(KB_INDEX).info()
    if int(info["num_docs"]) == 0:
        print("Knowledge base is empty, loading sample FAQs...")
        count = load_faqs(str(FAQS_PATH))
        print(f"Loaded {count} FAQs into the knowledge base")

    # Compared against the file's length rather than against zero: a
    # `num_docs == 0` check would mean that adding an exemplar to
    # data/guardrail_examples.json silently never loads, because a non-empty
    # index is never re-seeded. Changing an exemplar's *text* without changing
    # the count still needs `python scripts/load_kb.py`, which always reloads.
    with open(GUARDRAIL_EXAMPLES_PATH, encoding="utf-8") as f:
        expected_examples = len(jsonlib.load(f))

    guardrail_info = redis_client.ft(GUARDRAIL_INDEX).info()
    if int(guardrail_info["num_docs"]) != expected_examples:
        print("Guardrail exemplars are out of date, loading...")
        count = load_guardrail_examples(str(GUARDRAIL_EXAMPLES_PATH))
        print(f"Loaded {count} guardrail exemplars")

    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.exception_handler(openai.APIError)
async def llm_unreachable_handler(request: Request, exc: openai.APIError) -> JSONResponse:
    """Map any LLM failure to a 502 with a message the developer can act on.

    Catching the SDK's base error class covers both connection and status
    failures. Surfacing it here rather than swallowing it in the workflow is
    what keeps a dead LLM server from ever being cached as an answer — only a
    genuinely empty completion reaches `generation_failed`.
    """
    print(f"[llm] {type(exc).__name__}: {exc}")  # full detail server-side only
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "detail": (
                "The LLM backend is unreachable — is `local-openai.exe` running on :8080? "
                "Cache hits keep working without it; only misses need the LLM."
            )
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, _: None = Depends(enforce_rate_limit)):
    result = chat_workflow.invoke({"question": request.question})

    sources = [
        {"id": item["id"], "question": item["question"], "similarity": item["similarity"]}
        for item in result.get("context", [])
    ]

    return ChatResponse(
        answer=result["answer"],
        is_cached=result.get("is_cached", False),
        cache_similarity=result.get("cache_similarity"),
        sources=sources,
        guardrail=result.get("guardrail"),
        not_cached_reason=result.get("not_cached_reason"),
        cached_now=result.get("cached_now", False),
    )
