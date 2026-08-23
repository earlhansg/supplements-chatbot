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

    cache_similarity_threshold: float = 0.78
    cache_ttl_seconds: int = 86400

    kb_retrieval_k: int = 3


settings = Settings()
