/**
 * GET /api/status — liveness of both upstreams, for the header indicators.
 *
 * Kept separate from `/api/cache` so a Redis outage and a FastAPI outage are
 * distinguishable at a glance during a demo.
 */

import { pingBackend } from "@/lib/backend";
import { pingRedis } from "@/lib/redis";

export interface StatusResponse {
  backend: boolean;
  redis: boolean;
}

export async function GET(): Promise<Response> {
  const [backend, redis] = await Promise.all([pingBackend(), pingRedis()]);
  const body: StatusResponse = { backend, redis };
  return Response.json(body, { headers: { "Cache-Control": "no-store" } });
}
