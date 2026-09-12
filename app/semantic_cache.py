"""
Semantic response cache.

Every question that gets a fresh LLM-generated answer is stored as a
RedisJSON document at `cache:<uuid>`:

    {
      "query": "original user question",
      "answer": "LLM-generated answer",
      "embedding": [1536 floats]   # embedding of `query`
    }

with a Redis key TTL (`EXPIRE`) applied so entries age out automatically.
A RediSearch index `idx:cache` is built ON JSON over the `cache:*` prefix
with a VECTOR field (HNSW, COSINE) on `$.embedding`. Looking up the cache is
a KNN search for the single nearest neighbour; if its cosine similarity is
above `CACHE_SIMILARITY_THRESHOLD` we treat it as a cache hit, even if the
new question is worded differently from the one that was originally cached.

Both functions here take the question's embedding as a parameter rather than
computing it. The workflow embeds each question exactly once (in its
`embed_question` node) and shares that one symmetric vector between the
guardrail KNN and the cache KNN, so the safety layer costs a ~2 ms search
instead of a second ~15 ms encode.
"""

import uuid

from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from app.config import settings
from app.redis_client import redis_client
from app.vector_utils import floats_to_bytes

CACHE_PREFIX = "cache:"
CACHE_INDEX = "idx:cache"


def create_cache_index() -> None:
    schema = (
        TextField("$.query", as_name="query"),
        TextField("$.answer", as_name="answer"),
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
        redis_client.ft(CACHE_INDEX).create_index(
            schema,
            definition=IndexDefinition(prefix=[CACHE_PREFIX], index_type=IndexType.JSON),
        )
        print(f"Created RediSearch index '{CACHE_INDEX}'")
    except ResponseError as e:
        if "Index already exists" in str(e):
            print(f"RediSearch index '{CACHE_INDEX}' already exists, skipping creation")
        else:
            raise


def check_cache(query: str, embedding: list[float]) -> dict | None:
    """Look up the nearest cached question. Returns {answer, similarity} on a hit, else None.

    `query` is unused by the search itself — the vector is what matches — but is
    kept for signature symmetry with `save_cache`, which stores it, and so a
    future log line here has the question text to hand.
    """
    search_query = (
        Query("*=>[KNN 1 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("answer", "score")
        .paging(0, 1)
        .dialect(2)
    )

    results = redis_client.ft(CACHE_INDEX).search(
        search_query, query_params={"vec": floats_to_bytes(embedding)}
    )

    if not results.docs:
        return None

    doc = results.docs[0]
    similarity = 1 - float(doc.score)

    if similarity < settings.cache_similarity_threshold:
        return None

    return {"answer": doc.answer, "similarity": similarity}


def save_cache(query: str, answer: str, embedding: list[float]) -> None:
    key = f"{CACHE_PREFIX}{uuid.uuid4()}"
    redis_client.json().set(key, "$", {"query": query, "answer": answer, "embedding": embedding})
    redis_client.expire(key, settings.cache_ttl_seconds)
