"""
Threshold sweep for the semantic cache.

Measures what `CACHE_SIMILARITY_THRESHOLD` actually buys, over a labelled set
of question pairs in `data/eval_pairs.json`. Each pair is `{a, b, should_hit}`
where `a` is treated as already cached and `b` as the incoming question, and
**`should_hit` is true iff the cached answer for `a` fully answers `b`** — a
human judgement about answers, deliberately not a judgement about wording.
Labelling similarity would make this circular: similarity is the thing under
test.

Two modes are swept side by side:

    single   predicted hit iff sim >= t          the pre-Phase-3 behaviour
    banded   the real `classify_match()`         what ships today

`banded` calls `app.semantic_cache.classify_match` rather than re-implementing
the band rules. That is the whole reason the function was extracted: a harness
with its own copy of the rules keeps scoring the old logic after someone tunes
the real one, and the README's published table quietly becomes fiction.

Pair similarity is computed here as a plain **dot product** of the two
normalised vectors, which equals cosine because `app/embeddings.py` passes
`normalize_embeddings=True`. That is an exact number; the runtime's HNSW search
is approximate. On a corpus this small the two agree, but they are not the same
computation and this script does not pretend otherwise.

This harness reads `idx:kb` (for anchors) and **never writes to `idx:cache`**.
Polluting the demo cache with evaluation questions would corrupt the thing
being measured.

Usage:
    python scripts/eval_threshold.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.embeddings import generate_embeddings  # noqa: E402
from app.knowledge_base import anchor_for  # noqa: E402
from app.semantic_cache import classify_match  # noqa: E402

PAIRS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_pairs.json"

# Pinned, not derived from the data: a sweep whose row set moves with the corpus
# cannot be compared against a table published from an earlier run.
THRESHOLDS = [round(0.70 + 0.02 * i, 2) for i in range(13)]  # 0.70 .. 0.94

# The three bands `classify_match` considers good enough to serve. `rejected`
# and `miss` are the two that are not.
SERVED_BANDS = frozenset({"confident", "verified", "unverified"})


def confusion(predicted: list[bool], actual: list[bool]) -> dict:
    """Score one threshold. `false_hit_rate` is the metric this repo publishes."""
    tp = sum(p and a for p, a in zip(predicted, actual))
    fp = sum(p and not a for p, a in zip(predicted, actual))
    fn = sum(not p and a for p, a in zip(predicted, actual))
    tn = sum(not p and not a for p, a in zip(predicted, actual))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Of the pairs that should NOT have been served, the fraction that were.
    # Not 1 - precision: this one is relative to the negatives, so it does not
    # move when the positive class grows.
    false_hit_rate = fp / (fp + tn) if fp + tn else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_hit_rate": false_hit_rate,
    }


def markdown_table(rows: list[tuple[float, dict]]) -> str:
    """One row per threshold.

    Every float at 3 decimals, which is where the numbers are stable across
    ordinary torch / sentence-transformers float noise.
    """
    out = [
        "| Threshold | TP | FP | FN | TN | Precision | Recall | F1 | False-hit rate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t, m in rows:
        out.append(
            f"| {t:.3f} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {m['false_hit_rate']:.3f} |"
        )
    return "\n".join(out)


def main() -> None:
    pairs = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))

    # Every distinct question embedded exactly once, symmetrically (no is_query
    # prefix) — the same question <-> question comparison `check_cache` makes
    # against previously-cached questions.
    questions = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    vectors = dict(zip(questions, np.array(generate_embeddings(questions))))

    # Anchors cost a Redis round trip plus an is_query=True encode each, and the
    # sweep would otherwise repeat them once per threshold. `anchor_for(a)` is a
    # faithful stand-in for the anchor `save_cache` would have stored: the write
    # path takes its anchor from the same top-1 idx:kb match.
    anchors = {q: anchor_for(q) for q in questions}
    if all(anchor is None for anchor in anchors.values()):
        sys.exit(
            "idx:kb is empty, so every anchor is None and the banded mode would "
            "measure nothing. Seed it first: python scripts/load_kb.py"
        )

    actual = [p["should_hit"] for p in pairs]
    similarities = [float(vectors[p["a"]] @ vectors[p["b"]]) for p in pairs]

    single_rows, banded_rows = [], []
    original_threshold = settings.cache_similarity_threshold
    try:
        for t in THRESHOLDS:
            single_rows.append((t, confusion([s >= t for s in similarities], actual)))

            # classify_match reads the threshold off `settings` at call time, so
            # the sweep moves the setting rather than passing a value — that is
            # what keeps it scoring the shipped decision and not a copy of it.
            settings.cache_similarity_threshold = t
            served = []
            for pair, similarity in zip(pairs, similarities):
                entry = anchors[pair["a"]]
                decision = classify_match(
                    similarity,
                    entry_anchor=entry["id"] if entry else None,
                    entry_margin=entry["margin"] if entry else None,
                    # Bound as a default argument, not captured: a bare closure
                    # over the loop variable would resolve the last pair's
                    # anchor for every row.
                    resolve_incoming=lambda p=pair: anchors[p["b"]],
                )
                served.append(decision["band"] in SERVED_BANDS)
            banded_rows.append((t, confusion(served, actual)))
    finally:
        # The settings singleton is process-wide; leaving it moved would make a
        # later import in the same process silently score a different threshold.
        settings.cache_similarity_threshold = original_threshold

    positives = sum(actual)
    print("```bash")
    print("python scripts/eval_threshold.py")
    print("```")
    print()
    print(
        f"{len(pairs)} labelled pairs ({positives} should hit, "
        f"{len(pairs) - positives} should not), embedding model "
        f"`{settings.embedding_model}`."
    )
    print()
    print("**`false_hit_rate = FP / (FP + TN)`** — of the pairs that should *not* have")
    print("been served from cache, the fraction that were. A metric whose definition is")
    print("implicit is uncitable.")
    print()
    print("### Mode `single` — predicted hit iff `similarity >= t` (pre-Phase-3)")
    print()
    print(markdown_table(single_rows))
    print()
    print(
        f"### Mode `banded` — the shipped `classify_match()`, "
        f"`CACHE_HIT_THRESHOLD_HIGH={settings.cache_hit_threshold_high}`, "
        f"`CACHE_ANCHOR_MARGIN_MIN={settings.cache_anchor_margin_min}`"
    )
    print()
    print(markdown_table(banded_rows))
    print()

    # The verdict, in PRD R-1's own terms, at the threshold that actually ships.
    shipped = min(THRESHOLDS, key=lambda t: abs(t - original_threshold))
    s = dict(single_rows)[shipped]
    b = dict(banded_rows)[shipped]
    delta_fhr = b["false_hit_rate"] - s["false_hit_rate"]
    delta_recall = b["recall"] - s["recall"]
    passes = delta_fhr < 0 and delta_recall >= -0.05
    print(
        f"**Verdict at the shipped threshold {shipped:.2f}:** banded verification moves "
        f"the false-hit rate by {delta_fhr:+.3f} ({s['false_hit_rate']:.3f} -> "
        f"{b['false_hit_rate']:.3f}) and recall by {delta_recall:+.3f} "
        f"({s['recall']:.3f} -> {b['recall']:.3f}). PRD R-1 asks for a lower false-hit "
        f"rate at no worse than -0.05 recall: **{'PASS' if passes else 'FAIL'}**."
    )


if __name__ == "__main__":
    main()
