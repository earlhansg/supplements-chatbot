"""
Semantic guardrails — an input topic router and an output cache gate.

**Input guardrail.** A hand-rolled semantic topic router. Every labelled
exemplar is stored as a RedisJSON document at `guardrail:<id>`:

    {
      "id": "med-003",
      "label": "blocked:medical_advice",
      "action": "block",
      "text": "Will this supplement interact with my prescription medication?",
      "response": "I can't give medical advice. ...",
      "embedding": [768 floats]   # embedding of `text`, SYMMETRIC (no bge query prefix)
    }

A RediSearch index `idx:guardrail` is built ON JSON over the `guardrail:*`
prefix with a VECTOR field (HNSW, COSINE) on `$.embedding`, plus TAG fields on
label/action. Routing an incoming question is a KNN search for its single
nearest exemplar: if that exemplar's action is `block` and the cosine
similarity clears `GUARDRAIL_THRESHOLD`, the question is refused with the
exemplar's canned `response` — no LLM call, and `idx:cache` is never touched.

The `allowed:` exemplars are load-bearing, not decoration. Without them a
legitimate product question ("Is creatine safe?") has nothing nearby to land
on and drifts toward the nearest blocked cluster instead.

Exemplars are embedded **symmetrically** — `generate_embedding(text)` with no
`is_query=True` — because the comparison here is question <-> question, the
same shape the semantic cache does. That also lets one embedding of the
incoming question serve both the guardrail KNN and the cache KNN. Prefixing
either side would silently degrade match quality (see app/embeddings.py).

**Output guardrail.** `should_cache()` gates only the cache *write*. A
degraded answer still reaches the user; it just never gets enshrined for the
next `CACHE_TTL_SECONDS` where every paraphrase would be served it.
"""

import json as jsonlib

from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from app.config import settings
from app.embeddings import generate_embeddings
from app.redis_client import redis_client
from app.vector_utils import floats_to_bytes

GUARDRAIL_PREFIX = "guardrail:"
GUARDRAIL_INDEX = "idx:guardrail"

# Lowercase substrings marking an answer that refuses or hedges rather than
# answering. Deliberately a module constant and not a setting: these are
# prose-level policy that belongs next to the code applying it, and a list of
# patterns in .env would need JSON encoding and make the file unreadable.
REFUSAL_PATTERNS: tuple[str, ...] = (
    "i don't have information",
    "i cannot help",
    "i'm not able to",
    "as an ai",
    "i don't know",
)


def create_guardrail_index() -> None:
    schema = (
        TagField("$.label", as_name="label"),
        TagField("$.action", as_name="action"),
        TextField("$.text", as_name="text"),
        # `response` is indexed as TEXT purely so `return_fields` can resolve
        # it — on a JSON index that only works for indexed aliases, and an
        # unindexed path would need `return_field("$.response", as_field=...)`.
        # Same reason app/semantic_cache.py indexes `$.answer`: returned, never
        # text-searched.
        TextField("$.response", as_name="response"),
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
        redis_client.ft(GUARDRAIL_INDEX).create_index(
            schema,
            definition=IndexDefinition(
                prefix=[GUARDRAIL_PREFIX], index_type=IndexType.JSON
            ),
        )
        print(f"Created RediSearch index '{GUARDRAIL_INDEX}'")
    except ResponseError as e:
        if "Index already exists" in str(e):
            print(f"RediSearch index '{GUARDRAIL_INDEX}' already exists, skipping creation")
        else:
            raise


def load_guardrail_examples(examples_path: str) -> int:
    """Embed and store every labelled exemplar from a JSON file. Returns count loaded."""
    with open(examples_path, encoding="utf-8") as f:
        examples = jsonlib.load(f)

    # No `is_query=True`: exemplars are compared against the symmetric question
    # vector, so both sides must be embedded the same unprefixed way.
    embeddings = generate_embeddings([example["text"] for example in examples])

    pipeline = redis_client.pipeline(transaction=False)
    for example, embedding in zip(examples, embeddings):
        doc = {
            "id": example["id"],
            "label": example["label"],
            "action": example["action"],
            "text": example["text"],
            "response": example["response"],
            "embedding": embedding,
        }
        pipeline.json().set(f"{GUARDRAIL_PREFIX}{example['id']}", "$", doc)
    pipeline.execute()

    return len(examples)


def check_input(question: str, embedding: list[float]) -> dict | None:
    """Route a question against the exemplars. Returns a verdict dict on a block, else None.

    `embedding` is required rather than computed here: the workflow embeds the
    question once and shares that one symmetric vector with the cache lookup.
    An optional parameter that fell back to encoding would hide that property
    and let a caller silently pay for a second encode.
    """
    if not settings.guardrail_enabled:
        return None

    search_query = (
        Query("*=>[KNN 1 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("label", "action", "response", "score")  # never return the embedding
        .paging(0, 1)
        .dialect(2)
    )

    results = redis_client.ft(GUARDRAIL_INDEX).search(
        search_query, query_params={"vec": floats_to_bytes(embedding)}
    )

    if not results.docs:
        return None

    doc = results.docs[0]
    similarity = 1 - float(doc.score)  # COSINE distance -> similarity

    # A nearby `allow` exemplar is the whole point of having them, so only an
    # `action == "block"` neighbour can refuse — and only if it is close enough.
    if doc.action != "block" or similarity < settings.guardrail_threshold:
        return None

    return {
        "label": doc.label,
        "action": doc.action,
        "similarity": similarity,
        "response": doc.response,
    }


def should_cache(answer: str | None, context: list[dict]) -> tuple[bool, str | None]:
    """Decide whether an answer is fit to cache. Returns (True, None) or (False, reason).

    Runs before the write and *after* the response is already settled: every
    rule below suppresses the cache entry only, never the answer to the user.
    One bad generation should not be served to every paraphrase for a day.
    """
    # Checked first so a None answer never reaches len() below. The OpenAI SDK
    # types `message.content` as `str | None`, so this is genuinely reachable.
    if answer is None or not answer.strip():
        return False, "generation_failed"

    if len(answer.strip()) < settings.min_cacheable_answer_chars:
        return False, "too_short"

    lowered = answer.lower()
    if any(pattern in lowered for pattern in REFUSAL_PATTERNS):
        return False, "refusal"

    # An absent or empty context means nothing was retrieved to ground the
    # answer, which is ungrounded — never treat it as grounded by default.
    if not context or max(item["similarity"] for item in context) < settings.kb_grounding_floor:
        return False, "ungrounded"

    return True, None
