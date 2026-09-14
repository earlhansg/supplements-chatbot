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
    # The matched entry's hit total after this request. Set only on a hit, and
    # None even then for an entry written before cache documents carried a
    # `hits` field — those age out within CACHE_TTL_SECONDS.
    cached_hits: int | None = None


class DailyStats(BaseModel):
    """Counters for the current UTC day. Expire after STATS_DAILY_TTL_SECONDS."""

    hits: int
    misses: int


class StatsResponse(BaseModel):
    """What GET /stats reports — see app/metrics.py for where each number lives."""

    hits: int
    misses: int
    blocked: int
    # hits + misses. Blocked questions never reached the cache, so they are not
    # in the denominator of `hit_rate`.
    total: int
    hit_rate: float
    # Named for the reader rather than the data: it equals `hits`, and it is the
    # entire economic argument for the cache in one field.
    llm_calls_avoided: int
    cache_entries: int
    today: DailyStats
