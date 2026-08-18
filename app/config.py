from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central app configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Supplements Store Chatbot"

    openai_api_key: str
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    redis_url: str = "redis://127.0.0.1:6379"

    cache_similarity_threshold: float = 0.78
    cache_ttl_seconds: int = 86400

    kb_retrieval_k: int = 3


settings = Settings()
