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
    query_vector = generate_embedding(query, is_query=True)

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


def anchor_for(question: str) -> dict | None:
    """Resolve the FAQ a question anchors to, and how decisively.

    Returns {"id", "similarity", "margin"}, or None when the KB is empty.
    `margin` is the gap to the runner-up: a small gap means the top FAQ barely
    won and the anchor should not be trusted to veto anything.

    Takes the raw question rather than a vector, deliberately. This is a
    question -> passage comparison against passages indexed WITHOUT the bge
    prefix, so it must do its own `is_query=True` encode; accepting a vector
    would let a caller hand in the workflow's symmetric one, which would
    compile, run, return plausible ids, and silently degrade every anchor
    decision.
    """
    query_vector = generate_embedding(question, is_query=True)

    # KNN 2, not 1: the runner-up is the whole point — one score alone says
    # nothing about whether the winner actually won.
    search_query = (
        Query("*=>[KNN 2 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("score")  # the id arrives on doc.id; no FAQ text needed
        .paging(0, 2)
        .dialect(2)
    )

    results = redis_client.ft(KB_INDEX).search(
        search_query, query_params={"vec": floats_to_bytes(query_vector)}
    )

    if not results.docs:
        return None

    top = results.docs[0]
    similarity = 1 - float(top.score)
    # A single-document KB has no runner-up, so there is nothing the top FAQ
    # could be confused with: a margin of 1.0 says "maximally decisive" rather
    # than raising on a degenerate but legal corpus.
    margin = (
        similarity - (1 - float(results.docs[1].score)) if len(results.docs) > 1 else 1.0
    )

    return {
        "id": top.id.replace(KB_PREFIX, "", 1),
        "similarity": similarity,
        "margin": margin,
    }
