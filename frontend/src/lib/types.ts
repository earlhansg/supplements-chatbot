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

/** `app/schemas.py::GuardrailInfo` — the verdict when the input guardrail refused. */
export interface GuardrailInfo {
  label: string;
  similarity: number;
  action: string;
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
  /**
   * `app/schemas.py::ChatResponse.guardrail` — set only when the input
   * guardrail blocked the question, in which case `answer` is its canned
   * response and no LLM call was made.
   */
  guardrail?: GuardrailInfo | null;
  /**
   * `app/schemas.py::ChatResponse.not_cached_reason` — why the answer was not
   * written to the cache (`too_short`, `refusal`, `ungrounded`,
   * `generation_failed`). The answer still reached the user either way.
   */
  not_cached_reason?: string | null;
  /**
   * `app/schemas.py::ChatResponse.cached_now` — true only when this request's
   * answer was freshly written to the cache. False on a hit and on a block.
   */
  cached_now?: boolean;
  /**
   * `app/schemas.py::ChatResponse.cached_hits` — the matched entry's hit total
   * after this request. Set only on a hit, and null even then for an entry
   * written before cache documents carried a `hits` field.
   */
  cached_hits?: number | null;
}

/** `app/schemas.py::DailyStats` — counters for the current UTC day. */
export interface DailyStats {
  hits: number;
  misses: number;
}

/** `app/schemas.py::StatsResponse` — what `GET /stats` reports. */
export interface StatsResponse {
  hits: number;
  misses: number;
  blocked: number;
  /** `hits + misses`. Blocked questions never reached the cache, so they are
   *  not in the denominator of `hit_rate`. */
  total: number;
  hit_rate: number;
  /** The same number as `hits`, under the name that states the point. */
  llm_calls_avoided: number;
  cache_entries: number;
  today: DailyStats;
}

/**
 * One `cache:<uuid>` RedisJSON document, as surfaced by `/api/cache`.
 *
 * The stored document carries its own `created_at`, so `ageSeconds` is a real
 * age rather than an inference from the remaining TTL. Both `createdAt` and
 * `hits` are `null` for a document written before the backend added those
 * fields; such entries age out within `CACHE_TTL_SECONDS`.
 *
 * Note the deliberate casing split: the wire-contract types above keep
 * snake_case, while `CacheEntry` is this app's own view model assembled in
 * `lib/redis.ts` and uses camelCase throughout.
 */
export interface CacheEntry {
  key: string;
  query: string;
  answer: string;
  ttlSeconds: number | null;
  ageSeconds: number | null;
  /** Times this entry has been served from the cache. Null on a legacy document. */
  hits: number | null;
  /** Epoch seconds. Null on a legacy document. */
  createdAt: number | null;
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
  /**
   * Non-null when the input guardrail refused the question. That path ends the
   * graph before `check_cache` runs, so `isCached === false` here means "the
   * cache was never consulted" and not "the cache was searched and missed" —
   * the UI must render those two as different states.
   */
  guardrail: GuardrailInfo | null;
}

export interface ErrorMessage {
  id: string;
  role: "error";
  content: string;
  durationMs: number;
}

export type ChatMessage = UserMessage | AssistantMessage | ErrorMessage;

/** `blocked` is the guardrail path: no cache lookup, no LLM call, no KB retrieval. */
export type LogStatus = "hit" | "miss" | "blocked" | "error";

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
  /** Cache-entry similarity on a hit. Null on a miss, an error, and a block. */
  similarity: number | null;
  sourceCount: number;
  /** The refusing exemplar, on a `blocked` row only. Carries its own similarity. */
  guardrail: GuardrailInfo | null;
  error?: string;
}
