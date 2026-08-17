"""
Knowledge base storage + retrieval.

Each FAQ is stored as a RedisJSON document at `kb:<id>`:

    {
      "id": "faq-001",
      "category": "refunds",
      "question": "...",
      "answer": "...",
      "embedding": [1536 floats]   # embedding of "<question>\n<answer>"
    }

A RediSearch index `idx:kb` is built ON JSON over the `kb:*` prefix with a
VECTOR field (HNSW, COSINE) on `$.embedding`, plus TAG/TEXT fields so the KB
could also be filtered/browsed without going through the LLM at all.
"""

import json as jsonlib

from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from app.config import settings
from app.embeddings import generate_embedding, generate_embeddings
from app.redis_client import redis_client
from app.vector_utils import floats_to_bytes

KB_PREFIX = "kb:"
KB_INDEX = "idx:kb"


def create_kb_index() -> None:
    schema = (
        TextField("$.question", as_name="question"),
        TextField("$.answer", as_name="answer"),
        TagField("$.category", as_name="category"),
        VectorField(
            "$.embedding",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": settings.embedding_dim,
                "DISTANCE_METRIC": "COSINE",
            },
            as_name="embedding",
        ),
    )
    try:
        redis_client.ft(KB_INDEX).create_index(
            schema,
            definition=IndexDefinition(prefix=[KB_PREFIX], index_type=IndexType.JSON),
        )
        print(f"Created RediSearch index '{KB_INDEX}'")
    except ResponseError as e:
        if "Index already exists" in str(e):
            print(f"RediSearch index '{KB_INDEX}' already exists, skipping creation")
        else:
            raise


def load_faqs(faqs_path: str) -> int:
    """Embed and store every FAQ from a JSON file into Redis. Returns count loaded."""
    with open(faqs_path, encoding="utf-8") as f:
        faqs = jsonlib.load(f)

    texts = [f"{faq['question']}\n{faq['answer']}" for faq in faqs]
    embeddings = generate_embeddings(texts)

    pipeline = redis_client.pipeline(transaction=False)
    for faq, embedding in zip(faqs, embeddings):
        doc = {
            "id": faq["id"],
            "category": faq["category"],
            "question": faq["question"],
            "answer": faq["answer"],
            "embedding": embedding,
        }
        pipeline.json().set(f"{KB_PREFIX}{faq['id']}", "$", doc)
    pipeline.execute()

    return len(faqs)


def retrieve_context(query: str, k: int | None = None) -> list[dict]:
    """KNN vector search over the FAQ knowledge base, returns the top-k FAQ entries."""
    k = k or settings.kb_retrieval_k
    query_vector = generate_embedding(query)

    search_query = (
        Query(f"*=>[KNN {k} @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("question", "answer", "category", "score")
        .paging(0, k)
        .dialect(2)
    )

    results = redis_client.ft(KB_INDEX).search(
        search_query, query_params={"vec": floats_to_bytes(query_vector)}
    )

    return [
        {
            "id": doc.id.replace(KB_PREFIX, "", 1),
            "question": doc.question,
            "answer": doc.answer,
            "category": doc.category,
            "similarity": 1 - float(doc.score),
        }
        for doc in results.docs
    ]
