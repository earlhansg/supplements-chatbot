"""
Threshold sweep for the semantic input guardrail.

Tunes `GUARDRAIL_THRESHOLD`, which shipped in Phase 1 as an explicit
PLACEHOLDER, against `data/guardrail_eval.json` — a **held-out** set whose
text appears nowhere in `data/guardrail_examples.json`. That separation is the
whole validity of the number: an exemplar scores ~1.0 against itself, so a
sweep over the training set would recommend a flattering threshold that tells
you nothing about an unseen question.

The headline metric is **`false_block_rate = FP / (FP + TN)`** over the
`allow` questions — the fraction of legitimate product questions the guardrail
refuses. PRD R-2 cares about this one specifically: a demo that turns down
"Is creatine safe?" is worse than a demo with no guardrail at all.

Each question is searched **once** and the threshold rule applied in Python
afterwards. `check_input()` is deliberately not called in the loop: it reads
`settings.guardrail_threshold` internally, so sweeping through it would mean
one monkeypatch and one identical KNN per threshold. The block rule below
mirrors `app/guardrails.py` line for line instead — if that rule changes, this
must change with it.

Usage:
    python scripts/eval_guardrail.py
"""

import json
import sys
from pathlib import Path

from redis.commands.search.query import Query

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.embeddings import generate_embeddings  # noqa: E402
from app.guardrails import GUARDRAIL_INDEX  # noqa: E402
from app.redis_client import redis_client  # noqa: E402
from app.vector_utils import floats_to_bytes  # noqa: E402

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "guardrail_eval.json"

# Pinned, like the cache sweep's list, so two runs are comparable row for row.
# The range deliberately extends well below the shipped 0.72: the first run of
# this sweep put its optimum at the bottom edge of a narrower range, and a sweep
# that cannot see its own optimum reports the edge of its window rather than a
# measurement.
THRESHOLDS = [round(0.40 + 0.025 * i, 3) for i in range(21)]  # 0.400 .. 0.900


def nearest_exemplar(embedding: list[float]) -> tuple[str, float] | None:
    """The one KNN this sweep pays per question. Mirrors `check_input`'s query."""
    search_query = (
        Query("*=>[KNN 1 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("label", "action", "score")  # never return the embedding
        .paging(0, 1)
        .dialect(2)
    )
    results = redis_client.ft(GUARDRAIL_INDEX).search(
        search_query, query_params={"vec": floats_to_bytes(embedding)}
    )
    if not results.docs:
        return None

    doc = results.docs[0]
    return doc.action, 1 - float(doc.score)  # COSINE distance -> similarity


def confusion(predicted: list[bool], actual: list[bool]) -> dict:
    """Positive class is "blocked", so FP is a legitimate question turned away."""
    tp = sum(p and a for p, a in zip(predicted, actual))
    fp = sum(p and not a for p, a in zip(predicted, actual))
    fn = sum(not p and a for p, a in zip(predicted, actual))
    tn = sum(not p and not a for p, a in zip(predicted, actual))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Relative to the `allow` questions only — "how often does the guardrail
    # refuse someone it should have helped?".
    false_block_rate = fp / (fp + tn) if fp + tn else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_block_rate": false_block_rate,
    }


def main() -> None:
    if not settings.guardrail_enabled:
        sys.exit(
            "GUARDRAIL_ENABLED is false, so check_input() returns None for every "
            "question and this sweep would measure nothing. Enable it and re-run."
        )

    questions = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    # Symmetric, no is_query prefix — exemplars are indexed the same unprefixed
    # way, and it is the same vector the workflow shares with the cache lookup.
    embeddings = generate_embeddings([q["text"] for q in questions])

    nearest = [nearest_exemplar(e) for e in embeddings]
    if any(n is None for n in nearest):
        sys.exit(
            "idx:guardrail is empty, so no question has an exemplar to match. "
            "Seed it first: python scripts/load_kb.py"
        )

    actual = [q["expected"] == "block" for q in questions]

    rows = []
    for t in THRESHOLDS:
        # `check_input`'s rule, exactly: only a `block` neighbour can refuse, and
        # only when it is close enough.
        predicted = [action == "block" and similarity >= t for action, similarity in nearest]
        rows.append((t, confusion(predicted, actual)))

    blocks = sum(actual)
    print("```bash")
    print("python scripts/eval_guardrail.py")
    print("```")
    print()
    print(
        f"{len(questions)} held-out questions ({blocks} should block, "
        f"{len(questions) - blocks} should allow) against "
        f"{redis_client.ft(GUARDRAIL_INDEX).search(Query('*').paging(0, 0)).total} "
        f"seeded exemplars, embedding model `{settings.embedding_model}`. No text is "
        f"shared with `data/guardrail_examples.json`."
    )
    print()
    print("**`false_block_rate = FP / (FP + TN)`** — of the questions that should have")
    print("been allowed, the fraction the guardrail refused.")
    print()
    print("| Threshold | TP | FP | FN | TN | Precision | Recall | F1 | False-block rate |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t, m in rows:
        print(
            f"| {t:.3f} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {m['false_block_rate']:.3f} |"
        )
    print()

    # A threshold at or below the lowest similarity any question scored excludes
    # nothing: every verdict is then the nearest exemplar's `action` alone. Such
    # a row can post the best F1 on this corpus while describing a *disabled*
    # gate, so it is reported below and excluded from the recommendation —
    # recommending it would be recommending the knob's removal while calling it
    # a tuned value.
    floor = min(similarity for _, similarity in nearest)
    binding = [row for row in rows if row[0] > floor]

    # Highest F1 among the thresholds that bind; ties broken first by the lower
    # false-block rate (refusing a real customer is the failure this demo is
    # least willing to make) and then by the higher threshold, which blocks less.
    best_t, best = max(binding, key=lambda row: (row[1]["f1"], -row[1]["false_block_rate"], row[0]))
    shipped = min(THRESHOLDS, key=lambda t: abs(t - settings.guardrail_threshold))
    current = dict(rows)[shipped]
    unbound_t, unbound = max(rows, key=lambda row: (row[1]["f1"], -row[1]["false_block_rate"]))

    print(
        f"**Recommended threshold: {best_t:.3f}** — F1 {best['f1']:.3f}, precision "
        f"{best['precision']:.3f}, recall {best['recall']:.3f}, false-block rate "
        f"{best['false_block_rate']:.3f}, confusion matrix "
        f"TP={best['tp']} FP={best['fp']} FN={best['fn']} TN={best['tn']} "
        f"over n={len(questions)}."
    )
    print()
    print(
        f"**Currently shipped ({settings.guardrail_threshold}):** F1 {current['f1']:.3f}, "
        f"false-block rate {current['false_block_rate']:.3f}, "
        f"TP={current['tp']} FP={current['fp']} FN={current['fn']} TN={current['tn']}."
    )
    print()
    if unbound_t <= floor:
        print(
            f"Note: F1 is nominally higher ({unbound['f1']:.3f}) at t={unbound_t:.3f}, but the "
            f"lowest similarity any question scored is {floor:.3f}, so that threshold never "
            f"binds — it is the accuracy of the nearest-exemplar router with the gate switched "
            f"off, not a tuned threshold. It is excluded from the recommendation above."
        )
        print()
    print(
        f"n={len(questions)} is small: one question moving is worth roughly "
        f"{1 / len(questions):.1%} of the set, so treat this as calibration on this "
        f"corpus rather than a production-grade number. Re-run it after any embedding-"
        f"model swap or exemplar edit."
    )


if __name__ == "__main__":
    main()
