"""
Semantic response cache.

Every question that gets a fresh LLM-generated answer is stored as a
RedisJSON document at `cache:<uuid>`:

    {
      "query": "original user question",
      "answer": "LLM-generated answer",
      "embedding": [768 floats],   # embedding of `query`

      "created_at": 1757635200,    # epoch seconds, real write time
      "hits": 0,                   # JSON.NUMINCRBY'd on every hit
      "kb_anchor": "faq-003",      # top idx:kb match at write time
      "model": "BAAI/bge-base-en-v1.5"
    }

The four metadata fields are **not in the index schema**, and deliberately so.
RediSearch can return an unindexed JSON path via `RETURN $.path AS alias`, so
reading them costs nothing and adding them to the schema would buy nothing —
nothing filters or sorts on them inside Redis. `created_at` is what lets the
frontend show a real age instead of inferring one from the remaining TTL;
`hits` is what makes a popular entry visible; `model` makes an embedding-model
swap visible in the data rather than only in `.env`; `kb_anchor` records which
FAQ the answer was grounded in, for verification of borderline matches.

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

import time
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
    """Look up the nearest cached question. Returns {answer, similarity, key} on a hit, else None.

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

    # `doc.id` is the matched `cache:<uuid>` key. Returned so the caller can
    # bump that entry's own hit counter without re-running the search.
    return {"answer": doc.answer, "similarity": similarity, "key": doc.id}


def save_cache(
    query: str, answer: str, embedding: list[float], kb_anchor: str | None = None
) -> None:
    """Write one cache entry. `kb_anchor` defaults to None so the signature stays callable
    from anywhere; the workflow supplies the FAQ id it already retrieved."""
    key = f"{CACHE_PREFIX}{uuid.uuid4()}"
    redis_client.json().set(
        key,
        "$",
        {
            "query": query,
            "answer": answer,
            "embedding": embedding,
            # --- metadata, none of it indexed ---
            # Real write time, so the frontend no longer has to infer an age by
            # subtracting the remaining TTL from CACHE_TTL_SECONDS.
            "created_at": int(time.time()),
            "hits": 0,
            # Top idx:kb match at write time. Free here — the miss path already
            # retrieved it. Records which FAQ this answer was grounded in, so a
            # borderline cache match can later be confirmed against the same FAQ.
            "kb_anchor": kb_anchor,
            # Makes an embedding-model swap visible in the data rather than only
            # in .env: entries written by a different model are identifiable
            # instead of silently incomparable.
            "model": settings.embedding_model,
        },
    )
    redis_client.expire(key, settings.cache_ttl_seconds)
