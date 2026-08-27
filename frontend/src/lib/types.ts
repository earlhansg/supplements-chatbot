/**
 * Types mirroring the FastAPI backend contract.
 *
 * `ChatRequest` / `ChatResponse` are transcribed field-for-field from
 * `app/schemas.py` in the Python project — keep them in sync if that file
 * changes. Snake_case is preserved deliberately so the shape is obviously
 * the wire format and not something this app invented.
 */

/** `app/schemas.py::SourceFAQ` — a KB entry retrieved for a cache MISS. */
export interface SourceFAQ {
  id: string;
  question: string;
  similarity: number;
}

/** `app/schemas.py::ChatRequest` */
export interface ChatRequest {
  question: string;
}

/** `app/schemas.py::ChatResponse` */
export interface ChatResponse {
  answer: string;
  /** Authoritative cache hit/miss flag set by the LangGraph workflow. */
  is_cached: boolean;
  /** Cosine similarity of the matched cache entry. Non-null only on a HIT. */
  cache_similarity: number | null;
  /** Only populated on a MISS — KB retrieval is skipped entirely on a hit. */
  sources: SourceFAQ[];
}

/**
 * One `cache:<uuid>` RedisJSON document, as surfaced by `/api/cache`.
 *
 * The stored document is `{query, answer, embedding}` — there is no timestamp
 * field, so `ageSeconds` is derived from the key's remaining TTL against the
 * backend's configured `CACHE_TTL_SECONDS`. It is approximate by nature and is
 * `null` when the key carries no expiry.
 */
export interface CacheEntry {
  key: string;
  query: string;
  answer: string;
  ttlSeconds: number | null;
  ageSeconds: number | null;
}

export interface CacheListResponse {
  entries: CacheEntry[];
  /** Total docs in `idx:cache`, which may exceed `entries.length`. */
  total: number;
}

export interface ApiError {
  error: string;
  /** Set when the failure is a missing RediSearch index rather than an outage. */
  hint?: string;
}

/* ------------------------------------------------------------------ */
/* Client-only view models                                            */
/* ------------------------------------------------------------------ */

export interface UserMessage {
  id: string;
  role: "user";
  content: string;
}

export interface AssistantMessage {
  id: string;
  role: "assistant";
  content: string;
  isCached: boolean;
  similarity: number | null;
  sources: SourceFAQ[];
  durationMs: number;
}

export interface ErrorMessage {
  id: string;
  role: "error";
  content: string;
  durationMs: number;
}

export type ChatMessage = UserMessage | AssistantMessage | ErrorMessage;

export type LogStatus = "hit" | "miss" | "error";

/**
 * One row in the right-hand request log. Purely client-side session state —
 * never persisted, cleared on refresh.
 */
export interface LogEntry {
  id: string;
  question: string;
  status: LogStatus;
  /** Wall-clock duration of the `/api/chat` fetch, measured in the browser. */
  durationMs: number;
  /** Epoch ms, stamped when the response landed. */
  at: number;
  similarity: number | null;
  sourceCount: number;
  error?: string;
}
