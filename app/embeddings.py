from openai import OpenAI

from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)


def generate_embedding(text: str) -> list[float]:
    """Embed a single string with the configured OpenAI embedding model."""
    return generate_embeddings([text])[0]


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings in one API call."""
    response = _client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]
