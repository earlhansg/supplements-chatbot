"use client";

/**
 * Right panel — a session-only log of every chat request made from this tab.
 *
 * Deliberately not backed by anything: no polling, no backend log endpoint, no
 * persistence. State lives in React and is gone on refresh. Rows are appended
 * from the `/api/chat` response, and the duration is measured in the browser
 * around the fetch, because the backend does not report timing itself.
 */

import { useMemo } from "react";

import { cx } from "@/lib/cx";
import { formatClockTime, formatDuration, formatSimilarity } from "@/lib/format";
import type { LogEntry } from "@/lib/types";
import { ActivityIcon, BoltIcon, TrashIcon } from "@/components/icons";
import { Badge, EmptyState, IconButton, PanelBody, PanelHeader } from "@/components/ui";

export function RequestLogPanel({
  entries,
  onClear,
}: {
  entries: LogEntry[];
  onClear: () => void;
}) {
  const stats = useMemo(() => summarise(entries), [entries]);

  return (
    <section className="flex h-full min-h-0 flex-col bg-zinc-950">
      <PanelHeader
        icon={<ActivityIcon className="size-4" />}
        title="Request Log"
        subtitle={
          <span>
            {entries.length} {entries.length === 1 ? "request" : "requests"} · this session only
          </span>
        }
        actions={
          <IconButton label="Clear request log" onClick={onClear} disabled={entries.length === 0}>
            <TrashIcon className="size-4" />
          </IconButton>
        }
      />

      {stats ? <StatsStrip stats={stats} /> : null}

      <PanelBody>
        {entries.length === 0 ? (
          <EmptyState
            icon={<ActivityIcon className="size-5" />}
            title="No requests yet"
            description="Every question you ask is timed in the browser and logged here, newest first."
          />
        ) : (
          <ul className="divide-y divide-zinc-800/60">
            {entries.map((entry) => (
              <LogRow key={entry.id} entry={entry} />
            ))}
          </ul>
        )}
      </PanelBody>

      <footer className="shrink-0 border-t border-zinc-800/80 px-4 py-2 text-[11px] leading-relaxed text-zinc-600">
        Timings measured client-side around each request. Cleared on refresh.
      </footer>
    </section>
  );
}

/* ------------------------------------------------------------------ */

interface Stats {
  hits: number;
  misses: number;
  avgHitMs: number | null;
  avgMissMs: number | null;
}

/** The headline comparison: how much faster a hit is than a miss. */
function summarise(entries: LogEntry[]): Stats | null {
  const hits = entries.filter((entry) => entry.status === "hit");
  const misses = entries.filter((entry) => entry.status === "miss");
  if (hits.length === 0 && misses.length === 0) return null;

  const mean = (rows: LogEntry[]) =>
    rows.length === 0
      ? null
      : rows.reduce((total, row) => total + row.durationMs, 0) / rows.length;

  return {
    hits: hits.length,
    misses: misses.length,
    avgHitMs: mean(hits),
    avgMissMs: mean(misses),
  };
}

function StatsStrip({ stats }: { stats: Stats }) {
  const speedup =
    stats.avgHitMs !== null && stats.avgMissMs !== null && stats.avgHitMs > 0
      ? stats.avgMissMs / stats.avgHitMs
      : null;

  return (
    <div className="shrink-0 border-b border-zinc-800/80 bg-zinc-900/30 px-3 py-2.5">
      <div className="grid grid-cols-2 gap-2">
        <StatTile
          label={`${stats.hits} hit${stats.hits === 1 ? "" : "s"}`}
          value={stats.avgHitMs === null ? "—" : formatDuration(stats.avgHitMs)}
          tone="hit"
        />
        <StatTile
          label={`${stats.misses} miss${stats.misses === 1 ? "" : "es"}`}
          value={stats.avgMissMs === null ? "—" : formatDuration(stats.avgMissMs)}
          tone="miss"
        />
      </div>
      {speedup !== null ? (
        <p className="mt-2 text-center text-[11px] text-zinc-500">
          Cache hits are{" "}
          <span className="font-semibold text-emerald-400">{speedup.toFixed(0)}×</span> faster on
          average
        </p>
      ) : null}
    </div>
  );
}

function StatTile({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "hit" | "miss";
}) {
  return (
    <div
      className={cx(
        "rounded-lg border px-2.5 py-1.5",
        tone === "hit" ? "border-emerald-500/25 bg-emerald-500/5" : "border-amber-500/25 bg-amber-500/5",
      )}
    >
      <p className="text-[10px] tracking-wide text-zinc-500 uppercase">{label}</p>
      <p
        className={cx(
          "font-mono text-sm font-semibold tabular-nums",
          tone === "hit" ? "text-emerald-300" : "text-amber-300",
        )}
      >
        {value}
      </p>
      <p className="text-[10px] text-zinc-600">avg</p>
    </div>
  );
}

function LogRow({ entry }: { entry: LogEntry }) {
  return (
    <li className="px-4 py-3">
      <div className="flex items-center justify-between gap-2">
        {entry.status === "hit" ? (
          <Badge tone="hit">
            <BoltIcon className="size-3" />
            Cache Hit
          </Badge>
        ) : entry.status === "miss" ? (
          <Badge tone="miss">Cache Miss</Badge>
        ) : (
          <Badge tone="error">Error</Badge>
        )}

        <span
          className={cx(
            "font-mono text-xs font-semibold tabular-nums",
            entry.status === "hit"
              ? "text-emerald-300"
              : entry.status === "miss"
                ? "text-amber-300"
                : "text-rose-300",
          )}
        >
          {formatDuration(entry.durationMs)}
        </span>
      </div>

      <p className="mt-2 line-clamp-2 text-[13px] leading-snug text-zinc-300">{entry.question}</p>

      {entry.error ? (
        <p className="mt-1.5 line-clamp-2 text-[11px] leading-relaxed text-rose-400/80">
          {entry.error}
        </p>
      ) : null}

      <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-zinc-600">
        <time dateTime={new Date(entry.at).toISOString()} className="font-mono tabular-nums">
          {formatClockTime(entry.at)}
        </time>
        {entry.similarity !== null ? (
          <>
            <span className="text-zinc-800">·</span>
            <span title="Similarity to the matched cache entry">
              {formatSimilarity(entry.similarity)} match
            </span>
          </>
        ) : null}
        {entry.status === "miss" ? (
          <>
            <span className="text-zinc-800">·</span>
            <span>
              {entry.sourceCount} FAQ{entry.sourceCount === 1 ? "" : "s"} retrieved
            </span>
            <span className="text-zinc-800">·</span>
            <span className="text-zinc-600">answer cached</span>
          </>
        ) : null}
      </div>
    </li>
  );
}
