from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.config import settings
from app.knowledge_base import KB_INDEX, create_kb_index, load_faqs
from app.schemas import ChatRequest, ChatResponse
from app.semantic_cache import create_cache_index
from app.workflow import chat_workflow

FAQS_PATH = Path(__file__).resolve().parent.parent / "data" / "faqs.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_kb_index()
    create_cache_index()

    from app.redis_client import redis_client

    info = redis_client.ft(KB_INDEX).info()
    if int(info["num_docs"]) == 0:
        print("Knowledge base is empty, loading sample FAQs...")
        count = load_faqs(str(FAQS_PATH))
        print(f"Loaded {count} FAQs into the knowledge base")

    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
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
    )
