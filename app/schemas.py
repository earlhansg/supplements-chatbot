from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["How long does shipping take?"])


class SourceFAQ(BaseModel):
    id: str
    question: str
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    is_cached: bool
    cache_similarity: float | None = None
    sources: list[SourceFAQ] = []
