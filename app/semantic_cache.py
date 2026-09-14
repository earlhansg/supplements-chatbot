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
      "kb_anchor_margin": 0.125,   # how far that FAQ beat the runner-up
      "model": "BAAI/bge-base-en-v1.5"
    }

The five metadata fields are **not in the index schema**, and deliberately so.
RediSearch can return an unindexed JSON path via `RETURN $.path AS alias`, so
reading them costs nothing and adding them to the schema would buy nothing —
nothing filters or sorts on them inside Redis. `created_at` is what lets the
frontend show a real age instead of inferring one from the remaining TTL;
`hits` is what makes a popular entry visible; `model` makes an embedding-model
swap visible in the data rather than only in `.env`; `kb_anchor` records which
FAQ the answer was grounded in, for verification of borderline matches; and
`kb_anchor_margin` records how *decisively* it was that FAQ, so a borderline
match is never vetoed on the strength of a coin flip between two similar FAQs.

with a Redis key TTL (`EXPIRE`) applied so entries age out automatically.
A RediSearch index `idx:cache` is built ON JSON over the `cache:*` prefix
with a VECTOR field (HNSW, COSINE) on `$.embedding`. Looking up the cache is
a KNN search for the single nearest neighbour, whose cosine similarity places
it in one of three bands:

    sim >= CACHE_HIT_THRESHOLD_HIGH   confident  — served untouched
    CACHE_SIMILARITY_THRESHOLD <= sim grey       — served only if the incoming
                                                   question's KB anchor agrees
                                                   with the stored one, or if
                                                   either anchor is undecided
    sim <  CACHE_SIMILARITY_THRESHOLD miss       — answered from scratch

So a differently-worded question still hits, as before — but one that merely
shares a topic with a cached question is now rejected on the anchor rather
than served the wrong answer.

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
from app.knowledge_base import anchor_for
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


def _verdict(
    hit: bool,
    band: str | None = None,
    similarity: float | None = None,
    answer: str | None = None,
    key: str | None = None,
    entry_anchor: str | None = None,
    incoming_anchor: str | None = None,
) -> dict:
    """One shape for every outcome, so the caller never has to test for absent keys.

    The two anchor ids are set only where the comparison actually ran, and exist
    so the console line can say *which* FAQs disagreed — a rejection is far
    easier to read as `faq-005 != faq-006` than as a bare score.
    """
    return {
        "hit": hit,
        "band": band,
        "similarity": similarity,
        "answer": answer,
        "key": key,
        "entry_anchor": entry_anchor,
        "incoming_anchor": incoming_anchor,
    }


def check_cache(query: str, embedding: list[float]) -> dict:
    """Look up the nearest cached question and decide which band it falls in.

    Returns a `_verdict` dict — {hit, band, similarity, answer, key, and the two
    anchor ids} — in every case rather than None, so the caller can tell
    "nothing was close" apart from "something was close and was rejected". That
    near-miss score is the whole point of the grey band and must survive the
    return.

    `query` is not used by the search itself — the vector is what matches — but
    it is what `anchor_for()` re-embeds in the grey band, and `save_cache`
    stores it.
    """
    search_query = (
        Query("*=>[KNN 1 @embedding $vec AS score]")
        .sort_by("score")
        # The two anchor fields are metadata and NOT in the index schema, so a
        # bare alias resolves to nothing at all — silently. Only the explicit
        # `$.path AS alias` form reaches an unindexed JSON path.
        .return_fields("answer", "score")
        .return_field("$.kb_anchor", as_field="kb_anchor")
        .return_field("$.kb_anchor_margin", as_field="kb_anchor_margin")
        .paging(0, 1)
        .dialect(2)
    )

    results = redis_client.ft(CACHE_INDEX).search(
        search_query, query_params={"vec": floats_to_bytes(embedding)}
    )

    if not results.docs:
        return _verdict(hit=False)

    doc = results.docs[0]
    similarity = 1 - float(doc.score)

    if similarity < settings.cache_similarity_threshold:
        return _verdict(hit=False)

    # `doc.id` is the matched `cache:<uuid>` key. Carried so the caller can bump
    # that entry's own hit counter without re-running the search.
    served = {"similarity": similarity, "answer": doc.answer, "key": doc.id}

    # --- Band 1: confident. Close enough that verification could only cost time.
    # Returning here before any anchor work is a contract, not an optimisation:
    # the fast path must issue exactly one KNN, as it did before this band logic.
    if similarity >= settings.cache_hit_threshold_high:
        return _verdict(hit=True, band="confident", **served)

    # --- Band 2: grey. Similar enough to be a candidate, not similar enough to
    # trust on cosine alone.
    if not settings.cache_verify_grey_band:
        # Kill switch: the pre-verification behaviour exactly — serve it. The
        # band is still reported, so the response shape does not vary with the
        # flag and the two modes stay comparable.
        return _verdict(hit=True, band="unverified", **served)

    # Every FT.SEARCH value arrives as a string, and an absent JSON path drops
    # the attribute entirely (a JSON null yields None), so read both defensively.
    entry_anchor = getattr(doc, "kb_anchor", None)
    entry_margin_raw = getattr(doc, "kb_anchor_margin", None)

    if entry_anchor is None or entry_margin_raw is None:
        # An entry written before this phase has nothing to verify against. Treat
        # it as a miss rather than serving it unchecked: the re-answer overwrites
        # the gap with a document that carries both fields, so the cache heals
        # itself one entry at a time instead of staying permanently unverifiable.
        return _verdict(hit=False)

    entry_margin = float(entry_margin_raw)
    incoming = anchor_for(query)

    # An empty KB leaves nothing to anchor against — reachable only mid-reseed,
    # and not a reason to reject a match that already cleared the threshold.
    if incoming is None:
        return _verdict(hit=True, band="unverified", **served)

    # The margin guard. When either side's top FAQ barely beat its runner-up,
    # that side's anchor is a coin flip between two near-duplicate FAQs and
    # carries no information — it may not veto, and equally it may not confirm,
    # so an agreeing pair of untrusted anchors still serves as `unverified`.
    if (
        incoming["margin"] < settings.cache_anchor_margin_min
        or entry_margin < settings.cache_anchor_margin_min
    ):
        return _verdict(hit=True, band="unverified", **served)

    # Both anchors are decisive, so they mean something. Landing on the same FAQ
    # is what separates a rewording from a merely related question.
    anchors = {"entry_anchor": entry_anchor, "incoming_anchor": incoming["id"]}
    if incoming["id"] == entry_anchor:
        return _verdict(hit=True, band="verified", **served, **anchors)

    # Rejected: close on cosine, but about a different FAQ. The similarity rides
    # along so the caller can report the near miss instead of reporting silence.
    return _verdict(hit=False, band="rejected", similarity=similarity, **anchors)


def save_cache(
    query: str,
    answer: str,
    embedding: list[float],
    kb_anchor: str | None = None,
    kb_anchor_margin: float | None = None,
) -> None:
    """Write one cache entry. `kb_anchor` / `kb_anchor_margin` default to None so the
    signature stays callable from anywhere; the workflow supplies both from the KB
    search the miss path already ran."""
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
            # How far that FAQ beat the runner-up. Stored alongside the anchor
            # because the anchor alone cannot say whether it is trustworthy: two
            # near-duplicate FAQs produce a top-1 pick that is a coin flip, and
            # a coin flip must never be allowed to veto a later match.
            "kb_anchor_margin": kb_anchor_margin,
            # Makes an embedding-model swap visible in the data rather than only
            # in .env: entries written by a different model are identifiable
            # instead of silently incomparable.
            "model": settings.embedding_model,
        },
    )
    redis_client.expire(key, settings.cache_ttl_seconds)
