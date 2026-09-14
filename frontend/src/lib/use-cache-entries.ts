"use client";

/**
 * Polls `/api/cache` and `/api/stats` so the left panel reflects `idx:cache` as
 * it changes, including entries written by other sessions or by earlier runs.
 *
 * Client-side polling rather than a Server Component read because the data is
 * refetched on an interval — the Next.js docs call this out as one of the cases
 * that belongs in the browser. The first page load still gets its rows from the
 * server (see `app/page.tsx`), so the panel is never blank on arrival.
 *
 * Both requests share this one hook, and therefore one interval: two components
 * independently polling the same backend on their own timers is how a dashboard
 * ends up showing two different numbers for the same thing.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { CACHE_POLL_INTERVAL_MS, SLOW_REFRESH_MS } from "@/lib/constants";
import type { ApiError, CacheEntry, CacheListResponse, StatsResponse } from "@/lib/types";

export interface CacheFeed {
  entries: CacheEntry[];
  total: number;
  /** Backend counters, or null until the first successful read. `/api/stats`
   *  failing independently of `/api/cache` just hides the hit rate — it never
   *  blanks the rows. */
  stats: StatsResponse | null;
  /** Set only when there is nothing usable to show; a failed poll that follows a
   *  good one keeps the stale rows on screen instead of blanking the panel. */
  error: string | null;
  hint: string | null;
  /** True while a poll is in flight and rows are already on screen. */
  isRefreshing: boolean;
  /** True when the most recent poll failed but stale rows are still displayed. */
  isStale: boolean;
  /** Epoch ms of the last successful poll; null until the first one lands. */
  lastUpdated: number | null;
  refresh: () => void;
}

export function useCacheEntries(
  initialEntries: CacheEntry[],
  initialTotal: number,
  initialError: { error: string; hint?: string } | null,
): CacheFeed {
  const [entries, setEntries] = useState<CacheEntry[]>(initialEntries);
  const [total, setTotal] = useState(initialTotal);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [error, setError] = useState<string | null>(initialError?.error ?? null);
  const [hint, setHint] = useState<string | null>(initialError?.hint ?? null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isStale, setIsStale] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);

  // Guards against overlapping polls and against a slow response from a previous
  // interval landing after a newer one.
  const inFlight = useRef(false);
  const hasRows = useRef(initialEntries.length > 0);

  const load = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;

    // Deferred, not immediate: a poll that resolves inside one frame never sets
    // the flag at all, so the panel stays visually still instead of strobing
    // every CACHE_POLL_INTERVAL_MS. Only a genuinely slow read announces itself.
    const slowEnoughToShow = setTimeout(() => setIsRefreshing(true), SLOW_REFRESH_MS);

    try {
      // allSettled, not all: the entry list is the panel's reason to exist, so
      // a failing /api/stats must never take the rows down with it.
      const [cacheResult, statsResult] = await Promise.allSettled([
        fetch("/api/cache", { cache: "no-store" }),
        fetch("/api/stats", { cache: "no-store" }),
      ]);

      // Guarded rather than thrown: a failed stats read leaves the last known
      // counters on screen and must never reach the catch below, which speaks
      // only for the entry list.
      if (statsResult.status === "fulfilled" && statsResult.value.ok) {
        const stats = (await statsResult.value.json().catch(() => null)) as StatsResponse | null;
        if (stats) setStats(stats);
      }

      if (cacheResult.status === "rejected") throw cacheResult.reason;
      const response = cacheResult.value;

      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as ApiError | null;
        throw new Error(body?.error ?? `Request failed (${response.status})`, {
          cause: body?.hint,
        });
      }

      const data = (await response.json()) as CacheListResponse;
      setEntries(data.entries);
      setTotal(data.total);
      hasRows.current = data.entries.length > 0;
      setError(null);
      setHint(null);
      setIsStale(false);
      setLastUpdated(Date.now());
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Failed to read the cache.";
      const causeHint = caught instanceof Error && typeof caught.cause === "string" ? caught.cause : null;

      if (hasRows.current) {
        // Keep showing what we last read; just flag it as possibly out of date.
        setIsStale(true);
      } else {
        setError(message);
        setHint(causeHint);
      }
    } finally {
      clearTimeout(slowEnoughToShow);
      inFlight.current = false;
      // A no-op re-render-wise when the flag never went up, which is the
      // common case — React bails out on an unchanged state value.
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;

    const start = () => {
      if (interval !== null) return;
      void load();
      interval = setInterval(() => void load(), CACHE_POLL_INTERVAL_MS);
    };

    const stop = () => {
      if (interval === null) return;
      clearInterval(interval);
      interval = null;
    };

    // No point hammering Redis while nobody is looking at the panel.
    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") start();
      else stop();
    };

    onVisibilityChange();
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      document.removeEventListener("visibilitychange", onVisibilityChange);
      stop();
    };
  }, [load]);

  // Stable identity so callers can safely list it as an effect/callback dependency.
  const refresh = useCallback(() => void load(), [load]);

  return { entries, total, stats, error, hint, isRefreshing, isStale, lastUpdated, refresh };
}
