from pydantic import BaseModel, Field

from app.config import settings


class ChatRequest(BaseModel):
    # Bounds live here rather than in the handler body so FastAPI enforces and
    # documents them. Over-length input therefore comes back as a 422; the
    # Next.js boundary mirrors MAX_QUESTION_CHARS to give a 400 at the edge.
    question: str = Field(
        ...,
        min_length=1,
        max_length=settings.max_question_chars,
        examples=["How long does shipping take?"],
    )


class SourceFAQ(BaseModel):
    id: str
    question: str
    similarity: float


class GuardrailInfo(BaseModel):
    """The verdict from app/guardrails.py when a question was refused."""

    label: str
    similarity: float
    action: str


class ChatResponse(BaseModel):
    answer: str
    is_cached: bool
    cache_similarity: float | None = None
    sources: list[SourceFAQ] = []
    # Every field below is optional with a default, so an older client keeps
    # working unchanged.
    #
    # Set only when the input guardrail blocked the question.
    guardrail: GuardrailInfo | None = None
    # Why the answer was not written to the cache, on a miss the output
    # guardrail rejected. The answer still reached the user either way.
    not_cached_reason: str | None = None
    # True only when this request's answer was freshly cached. Defaults to
    # False, which is already correct for a hit and for a blocked question.
    cached_now: bool = False
