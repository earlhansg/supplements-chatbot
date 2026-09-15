from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central app configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Supplements Store Chatbot"

    # Hosted OpenAI (used by app/llm.py). Optional: leave empty when running
    # against the local OpenAI-compatible server via app/llm_local.py.
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"

    # Which backend app/llm_factory.py resolves `generate_answer` from:
    # "local" (app/llm_local.py, an OpenAI-compatible server on
    # LOCAL_LLM_BASE_URL) or "openai" (app/llm.py, the hosted API). Any other
    # value raises at import rather than falling back. Run GET /v1/models on a
    # local server to see which model ids it accepts.
    llm_backend: str = "openai"

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
    # Measured: `python scripts/eval_threshold.py` over data/eval_pairs.json (23
    # labelled pairs). Sweeping this value holds F1 flat at 0.880 across the
    # whole band 0.03-0.10, so 0.05 was TESTED AND KEPT rather than merely
    # inherited — it sits in the middle of the widest optimum, not on an edge.
    # Either side of that band is measurably worse on this corpus: at <=0.02 the
    # guard stops protecting the refunds pair whose anchors are a coin flip
    # (margins 0.021 / 0.032) and recall falls 0.846 -> 0.769; at >=0.15 the
    # guard swallows the shipping anchors too (margins 0.125 / 0.108) and the
    # 0.809 false hit is served again, doubling the false-hit rate to 0.200.
    #
    # Raise it to trust the anchor less (more matches served unverified), lower
    # it to trust it more (more matches rejected on a narrow anchor win).
    # Corpus- AND model-dependent: re-run that sweep after editing data/faqs.json
    # or swapping the embedding model.
    cache_anchor_margin_min: float = 0.05

    kb_retrieval_k: int = 3

    # Semantic input guardrail (app/guardrails.py).
    #
    # Measured: `python scripts/eval_guardrail.py` over data/guardrail_eval.json
    # (24 held-out questions, no text shared with the exemplars) picks 0.55 as
    # the F1-maximising block threshold among the thresholds that actually bind.
    # The 0.72 this replaced was badly miscalibrated: it sat above the similarity
    # of 9 of the 12 questions that should have been blocked, scoring F1 0.400 to
    # 0.55's 0.870 and recall 0.250 to 0.833. The cost of the move is one false
    # block out of 12 allow questions ("Can I take this on an empty stomach?",
    # whose nearest exemplar is a medical one at 0.687) — a false-block rate of
    # 0.083, up from 0.000.
    #
    # n=24, so one question is worth ~4% of the set: this is calibration on this
    # corpus, not a production-grade number. Like `cache_similarity_threshold` it
    # is embedding-model-dependent — bge's cosine similarities run lower than
    # OpenAI's — so re-run that sweep rather than carrying the value across a
    # model swap. Raise it to block less (more false allows), lower it to block
    # more (more false blocks on legitimate product questions).
    guardrail_enabled: bool = True
    guardrail_threshold: float = 0.55

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
