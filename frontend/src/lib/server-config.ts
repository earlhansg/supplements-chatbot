/**
 * Server-only configuration. Never import this from a Client Component —
 * `REDIS_URL` and `BACKEND_URL` are deliberately not `NEXT_PUBLIC_`, so the
 * browser only ever talks to this app's own Route Handlers.
 */

function num(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export const serverConfig = {
  /** uvicorn host for the FastAPI app (`app/main.py`). */
  backendUrl: (process.env.BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/$/, ""),

  redisUrl: process.env.REDIS_URL ?? "redis://127.0.0.1:6379",

  /**
   * Must match `CACHE_TTL_SECONDS` in the backend's `.env`. Used only to turn a
   * key's remaining TTL back into an approximate age, since cache documents
   * carry no timestamp of their own.
   */
  cacheTtlSeconds: num(process.env.CACHE_TTL_SECONDS, 86400),

  /** Mirrors the backend's `CACHE_SIMILARITY_THRESHOLD`; displayed in the UI. */
  cacheSimilarityThreshold: num(process.env.CACHE_SIMILARITY_THRESHOLD, 0.78),

  /** How many cache entries the left panel lists at most. */
  cacheListLimit: num(process.env.CACHE_LIST_LIMIT, 100),

  /**
   * Generous ceiling: the backend proxies to a local LLM whose own timeout is
   * `LOCAL_LLM_TIMEOUT_SECONDS` (120s by default), so a cache miss can legitimately
   * take minutes on cold hardware.
   */
  chatTimeoutMs: num(process.env.CHAT_TIMEOUT_MS, 150_000),
} as const;

/** Config values that are safe to hand down to Client Components. */
export interface PublicConfig {
  cacheSimilarityThreshold: number;
  cacheTtlSeconds: number;
}

export const publicConfig: PublicConfig = {
  cacheSimilarityThreshold: serverConfig.cacheSimilarityThreshold,
  cacheTtlSeconds: serverConfig.cacheTtlSeconds,
};
