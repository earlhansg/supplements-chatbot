/**
 * GET /api/stats — thin proxy to the FastAPI `/stats` endpoint.
 *
 * Polled alongside `/api/cache` by the left panel's feed hook, so the hit rate
 * in the cache-panel header moves while the app is being used.
 *
 * No `maxDuration` here: that exists on `/api/chat` only because a cache miss
 * calls the LLM. A stats read is a handful of Redis reads and a count.
 */

import { BackendError, fetchStats } from "@/lib/backend";
import type { ApiError } from "@/lib/types";

export async function GET(): Promise<Response> {
  try {
    const stats = await fetchStats();
    return Response.json(stats, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    if (error instanceof BackendError) {
      // 4xx from the backend is the caller's fault; anything else is an upstream outage.
      const status = error.status >= 400 && error.status < 500 ? error.status : 502;
      const body: ApiError = { error: error.message };
      return Response.json(body, { status, headers: { "Cache-Control": "no-store" } });
    }
    console.error("[api/stats]", error);
    const body: ApiError = { error: "Unexpected error while reading the cache metrics." };
    return Response.json(body, { status: 500, headers: { "Cache-Control": "no-store" } });
  }
}
