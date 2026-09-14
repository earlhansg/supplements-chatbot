/** Presentation helpers shared by the three panels. Safe on both server and client. */

/** `41 ms` / `1.24 s` — sub-second stays in ms so cache hits read as instant. */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

/**
 * Compact relative age: `just now`, `4m ago`, `3h ago`, `2d ago`.
 *
 * `null` means the document carries no `created_at` — it was written before the
 * backend added that field — not that the key has no expiry.
 */
export function formatAge(seconds: number | null): string {
  if (seconds === null) return "age unknown";
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

/** Time remaining, e.g. `under a minute`, `12m`, `20h`, `1d`. */
export function formatCountdown(seconds: number): string {
  if (seconds < 60) return "under a minute";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
  return `${Math.floor(seconds / 86400)}d`;
}

/** Local wall-clock time for log rows, e.g. `14:07:33`. */
export function formatClockTime(epochMs: number): string {
  return new Date(epochMs).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

/** A 0-1 fraction as a percentage: `0.9412` -> `94.1%`. */
export function formatPercent(fraction: number): string {
  return `${(fraction * 100).toFixed(1)}%`;
}

/** Cosine similarity as a percentage. Same rendering as any other fraction —
 *  named separately because that is what call sites at the chat panel mean. */
export function formatSimilarity(similarity: number): string {
  return formatPercent(similarity);
}

/**
 * A sentence explaining what a `cache_band` value means, for the badge's
 * `title`. The badge itself already carries the word — this is the paragraph a
 * reader needs once to learn what the word is claiming, and the reason the band
 * needs no colour of its own.
 */
export function bandExplanation(band: string | null): string {
  switch (band) {
    case "confident":
      return "Similarity alone was high enough to serve this answer — no verification lookup was issued.";
    case "verified":
      return "A borderline match, confirmed: this question and the cached one resolve to the same FAQ.";
    case "unverified":
      return "A borderline match, served unchecked: the KB anchor was too close a call to confirm or to veto it.";
    case "rejected":
      return "A borderline match, declined: it resolves to a different FAQ, so this question was answered from scratch.";
    default:
      return "Cosine similarity between this question and the nearest cached one.";
  }
}

/**
 * `blocked:medical_advice` -> `Medical advice`.
 *
 * Guardrail labels are stored as `<action>:<topic>` (see `app/guardrails.py`).
 * The action half is dropped because the badge already says it was blocked;
 * only the topic tells the reader anything new.
 */
export function formatGuardrailLabel(label: string): string {
  const topic = label.slice(label.indexOf(":") + 1).replace(/_/g, " ");
  return topic.charAt(0).toUpperCase() + topic.slice(1);
}

/** `cache:8f3ad2e1-...-9c` -> `cache:8f3ad2e1` — enough to identify a row without wrapping. */
export function shortKey(key: string): string {
  const separator = key.indexOf(":");
  if (separator === -1) return key;
  return `${key.slice(0, separator + 1)}${key.slice(separator + 1, separator + 9)}`;
}
