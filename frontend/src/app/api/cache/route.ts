/**
 * GET /api/cache — lists what currently lives in the `idx:cache` RediSearch index.
 *
 * Polled by the left panel so entries written by other sessions (or by earlier
 * runs, since cache keys survive with a 24h TTL) show up without a refresh.
 */

import { CacheUnavailableError, listCachedEntries } from "@/lib/redis";
import type { ApiError } from "@/lib/types";

export async function GET(): Promise<Response> {
  try {
    const result = await listCachedEntries();
    return Response.json(result, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    if (error instanceof CacheUnavailableError) {
      const body: ApiError = { error: error.message, hint: error.hint };
      return Response.json(body, { status: 503, headers: { "Cache-Control": "no-store" } });
    }
    console.error("[api/cache]", error);
    const body: ApiError = { error: "Unexpected error while reading the semantic cache." };
    return Response.json(body, { status: 500, headers: { "Cache-Control": "no-store" } });
  }
}
