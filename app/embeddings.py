from sentence_transformers import SentenceTransformer

from app.config import settings

_model = SentenceTransformer(settings.embedding_model)

# bge-style models are trained asymmetrically: prefixing the *query* side (but
# not the passage/document side) with this instruction measurably improves
# retrieval quality. See https://huggingface.co/BAAI/bge-base-en-v1.5
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def generate_embedding(text: str, is_query: bool = False) -> list[float]:
    """Embed a single string with the configured local embedding model."""
    return generate_embeddings([text], is_query=is_query)[0]


def generate_embeddings(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Embed a batch of strings locally, in one call."""
    if is_query:
        texts = [QUERY_INSTRUCTION + text for text in texts]
    embeddings = _model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()
