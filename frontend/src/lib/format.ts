/** Presentation helpers shared by the three panels. Safe on both server and client. */

/** `41 ms` / `1.24 s` — sub-second stays in ms so cache hits read as instant. */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

/** Compact relative age: `just now`, `4m ago`, `3h ago`, `2d ago`. */
export function formatAge(seconds: number | null): string {
  if (seconds === null) return "no expiry";
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

/** Cosine similarity as a percentage: `0.9412` -> `94.1%`. */
export function formatSimilarity(similarity: number): string {
  return `${(similarity * 100).toFixed(1)}%`;
}

/** `cache:8f3ad2e1-...-9c` -> `cache:8f3ad2e1` — enough to identify a row without wrapping. */
export function shortKey(key: string): string {
  const separator = key.indexOf(":");
  if (separator === -1) return key;
  return `${key.slice(0, separator + 1)}${key.slice(separator + 1, separator + 9)}`;
}
