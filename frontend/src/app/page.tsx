/**
 * Server Component shell.
 *
 * The first list of cache entries is read straight from Redis here rather than
 * through `/api/cache` — the Next.js docs are explicit that a Server Component
 * should hit its data source directly instead of round-tripping through its own
 * Route Handler. The client takes over polling from that point on.
 */

import { Dashboard } from "@/components/Dashboard";
import { listCachedEntries, CacheUnavailableError } from "@/lib/redis";
import { publicConfig } from "@/lib/server-config";
import type { CacheEntry } from "@/lib/types";

// Redis is read at request time, so there is nothing to prerender at build time
// (and no Redis to connect to during a build).
export const dynamic = "force-dynamic";

export default async function Page() {
  let entries: CacheEntry[] = [];
  let total = 0;
  let cacheError: { error: string; hint?: string } | null = null;

  try {
    const result = await listCachedEntries();
    entries = result.entries;
    total = result.total;
  } catch (error) {
    // A cold Redis or a backend that has never started is an expected state on
    // first run, not a crash — hand it to the panel to explain.
    cacheError =
      error instanceof CacheUnavailableError
        ? { error: error.message, hint: error.hint }
        : { error: "Could not read the semantic cache." };
  }

  return (
    <Dashboard
      initialEntries={entries}
      initialTotal={total}
      initialCacheError={cacheError}
      config={publicConfig}
    />
  );
}
