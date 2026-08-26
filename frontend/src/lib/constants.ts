/** How often the left panel re-reads `idx:cache`. Paused while the tab is hidden. */
export const CACHE_POLL_INTERVAL_MS = 4_000;

/**
 * Starter questions for the empty chat, ordered so the demo works top to bottom:
 * ask an original, then the paraphrase below it.
 *
 * The paraphrases are not guesses — each was scored against its original with
 * the backend's own embedding model (BAAI/bge-base-en-v1.5, the same symmetric
 * embedding `check_cache` uses) and clears the 0.78 hit threshold:
 *
 *   "How do refunds work?"                  vs refund policy   -> 0.873
 *   "How many days until my order arrives?" vs shipping time   -> 0.817
 *
 * Worth knowing when picking your own: plausible-sounding paraphrases can still
 * fall short. "Can I get my money back on an unopened tub?" scores only 0.673
 * against "What is your refund policy?" and comes back as a miss.
 */
export const SUGGESTED_QUESTIONS = [
  "What is your refund policy?",
  "How do refunds work?",
  "How long does shipping take?",
  "How many days until my order arrives?",
] as const;
