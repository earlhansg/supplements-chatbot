/**
 * Read-only access to the backend's semantic cache.
 *
 * The Python backend never exposes `idx:cache` over HTTP — `check_cache()` in
 * `app/semantic_cache.py` is internal to the LangGraph workflow — so this app
 * reads the index directly. It only ever issues `FT.SEARCH` and `TTL`; nothing
 * here writes to or mutates Redis.
 *
 * Index/document layout (see `app/semantic_cache.py`):
 *   key    cache:<uuid>            RedisJSON, EXPIRE'd to CACHE_TTL_SECONDS
 *   doc    { query, answer, embedding: number[768] }
 *   index  idx:cache  ON JSON  PREFIX cache:  with aliases query/answer/embedding
 */

import { createClient, type RedisClientType } from "redis";

import { serverConfig } from "@/lib/server-config";
import type { CacheEntry, CacheListResponse } from "@/lib/types";

const CACHE_INDEX = "idx:cache";

/** Thrown for conditions the UI can explain to the user rather than just "500". */
export class CacheUnavailableError extends Error {
  readonly hint?: string;

  constructor(message: string, hint?: string) {
    super(message);
    this.name = "CacheUnavailableError";
    this.hint = hint;
  }
}

/**
 * A single connection reused across requests, stashed on `globalThis` so Turbopack
 * hot reloads in dev don't leak a new client on every edit.
 */
const globalForRedis = globalThis as typeof globalThis & {
  __supplementsRedis?: Promise<RedisClientType>;
};

function connect(): Promise<RedisClientType> {
  const client: RedisClientType = createClient({ url: serverConfig.redisUrl });

  // node-redis emits 'error' on every failed reconnect attempt. Without a listener
  // those become unhandled exceptions and take the dev server down.
  client.on("error", (error: Error) => {
    console.error("[redis]", error.message);
  });

  return client.connect().catch((error: unknown) => {
    // Let the next request retry from scratch instead of caching a dead promise.
    globalForRedis.__supplementsRedis = undefined;
    throw error;
  });
}

function getClient(): Promise<RedisClientType> {
  globalForRedis.__supplementsRedis ??= connect();
  return globalForRedis.__supplementsRedis;
}

function isMissingIndex(error: unknown): boolean {
  return error instanceof Error && /no such index/i.test(error.message);
}

/**
 * Cache documents have no `created_at`, so age is inferred from how much of the
 * TTL window is left. `TTL` returns -1 for a key with no expiry and -2 if it
 * vanished between the search and this call; both yield a `null` age rather than
 * a made-up number.
 */
function ageFromTtl(ttl: number): { ttlSeconds: number | null; ageSeconds: number | null } {
  if (ttl < 0) return { ttlSeconds: null, ageSeconds: null };
  const age = serverConfig.cacheTtlSeconds - ttl;
  return { ttlSeconds: ttl, ageSeconds: age >= 0 ? age : null };
}

/**
 * List cached Q&A pairs, newest first.
 *
 * `RETURN` is restricted to query/answer on purpose: without it every document
 * would drag its 768-float embedding across the wire.
 */
export async function listCachedEntries(
  limit: number = serverConfig.cacheListLimit,
): Promise<CacheListResponse> {
  let client: RedisClientType;
  try {
    client = await getClient();
  } catch {
    throw new CacheUnavailableError(
      `Cannot reach Redis at ${serverConfig.redisUrl}.`,
      "Start it with `docker compose up -d` in the backend project.",
    );
  }

  let reply;
  try {
    reply = await client.ft.search(CACHE_INDEX, "*", {
      RETURN: ["query", "answer"],
      LIMIT: { from: 0, size: limit },
      DIALECT: 2,
    });
  } catch (error) {
    if (isMissingIndex(error)) {
      throw new CacheUnavailableError(
        `RediSearch index '${CACHE_INDEX}' does not exist yet.`,
        "Start the FastAPI backend once — it creates the index on startup.",
      );
    }
    throw new CacheUnavailableError(
      error instanceof Error ? error.message : "Redis search failed.",
    );
  }

  // node-redis pipelines concurrent commands on the shared connection, so this
  // is one round trip's worth of latency rather than one per key.
  const ttls = await Promise.all(reply.documents.map((doc) => client.ttl(doc.id)));

  const entries: CacheEntry[] = reply.documents.map((doc, index) => {
    const value = doc.value as { query?: unknown; answer?: unknown };
    return {
      key: doc.id,
      query: typeof value.query === "string" ? value.query : "(missing query)",
      answer: typeof value.answer === "string" ? value.answer : "(missing answer)",
      ...ageFromTtl(ttls[index] ?? -1),
    };
  });

  // Most TTL left == most recently written. Entries without a TTL sort last.
  entries.sort((a, b) => (b.ttlSeconds ?? -1) - (a.ttlSeconds ?? -1));

  return { entries, total: reply.total };
}

/** Lightweight liveness probe for the status indicator in the header. */
export async function pingRedis(): Promise<boolean> {
  try {
    const client = await getClient();
    await client.ping();
    return true;
  } catch {
    return false;
  }
}
