from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central app configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Supplements Store Chatbot"

    # Hosted OpenAI (used by app/llm.py). Optional: leave empty when running
    # against the local OpenAI-compatible server via app/llm_local.py.
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"

    # Local OpenAI-compatible server (used by app/llm_local.py). The server
    # ignores credentials, so the key is a placeholder.
    local_llm_base_url: str = "http://127.0.0.1:8080/v1"
    local_llm_api_key: str = "unused"
    local_llm_timeout_seconds: float = 120.0
    local_chat_model: str = "sonnet"

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    redis_url: str = "redis://127.0.0.1:6379"

    # The LOWER bound of the grey band, not a simple hit line any more: at or
    # above it a match is a candidate, but between here and
    # `cache_hit_threshold_high` it must still survive anchor verification.
    cache_similarity_threshold: float = 0.78
    cache_ttl_seconds: int = 86400

    # ---- Grey-band verification (app/semantic_cache.py) ----
    # At or above this the nearest cached question is close enough to serve
    # untouched, and no anchor lookup is issued at all. Raise it to verify more
    # matches (slower hits, fewer wrong answers), lower it to verify fewer.
    # Embedding-model-dependent for the same reason as
    # `cache_similarity_threshold` — never carry this number across a model swap.
    cache_hit_threshold_high: float = 0.90
    # Master switch for the grey band's anchor check. false restores the
    # pre-verification behaviour exactly: everything at or above
    # `cache_similarity_threshold` is served. `cache_band` stays populated
    # either way, so the response shape never varies with the flag.
    cache_verify_grey_band: bool = True
    # How far the top idx:kb FAQ must beat its runner-up before the anchor is
    # allowed to veto a match. Below this gap the top-1 choice is a coin flip
    # between two near-duplicate FAQs, and an undecided anchor must not reject a
    # match that already cleared the lower threshold.
    #
    # 0.05 is derived from 10 FAQs and 6 question pairs on this corpus — that is
    # evidence, not calibration. It is a PLACEHOLDER pending the labelled sweep
    # of a later phase, exactly like `guardrail_threshold`. Raise it to trust the
    # anchor less (more matches served unverified), lower it to trust it more
    # (more matches rejected on a narrow anchor win). Corpus- AND
    # model-dependent: re-measure after editing data/faqs.json.
    cache_anchor_margin_min: float = 0.05

    kb_retrieval_k: int = 3

    # Semantic input guardrail (app/guardrails.py). `guardrail_threshold` is a
    # PLACEHOLDER, not a measured value — the evaluation harness that would tune
    # it against a labelled set is a later phase. Like
    # `cache_similarity_threshold` it is embedding-model-dependent: bge's cosine
    # similarities run lower than OpenAI's, so never copy this number across a
    # model swap. Raise it to block less (more false allows), lower it to block
    # more (more false blocks on legitimate product questions).
    guardrail_enabled: bool = True
    guardrail_threshold: float = 0.72

    # Output guardrail — the gate on the cache *write*, not on the response.
    # An answer shorter than this is almost certainly truncated or a one-line
    # dodge, so it is answered but never enshrined for CACHE_TTL_SECONDS.
    min_cacheable_answer_chars: int = 40
    # Floor on the best `idx:kb` similarity for an answer to count as grounded.
    # Below it the LLM improvised beyond the FAQs, so the answer is served but
    # not cached. Also model-dependent — re-check on an embedding model swap.
    kb_grounding_floor: float = 0.50
    # Upper bound on a question, enforced by Pydantic on ChatRequest. Long
    # enough for a real support question, short enough that a pasted document
    # cannot drive an expensive embed + LLM call.
    max_question_chars: int = 500

    # Fixed-window rate limiting (app/rate_limit.py). Bounds LLM spend: every
    # cache miss is a multi-second generation. 30 requests per 60s is generous
    # for a single developer exercising the demo by hand.
    rate_limit_enabled: bool = True
    rate_limit_max_requests: int = 30
    rate_limit_window_seconds: int = 60

    # Per-day stats counters (app/metrics.py). Long enough to show a month of
    # usage in a demo, short enough that the keyspace cannot grow without bound.
    stats_daily_ttl_seconds: int = 2_592_000  # 30 days


settings = Settings()
