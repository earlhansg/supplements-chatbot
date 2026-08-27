"use client";

/**
 * Left panel — everything currently sitting in the backend's semantic cache.
 *
 * This is the panel that makes the point during a demo: ask a question, watch a
 * new row appear here, then ask a paraphrase of it and watch the chat come back
 * as a HIT against that row.
 */

import { useState } from "react";

import { cx } from "@/lib/cx";
import { formatAge, formatCountdown, shortKey } from "@/lib/format";
import type { CacheFeed } from "@/lib/use-cache-entries";
import type { CacheEntry } from "@/lib/types";
import { DatabaseIcon, RefreshIcon, SearchIcon } from "@/components/icons";
import {
  Badge,
  EmptyState,
  ErrorState,
  IconButton,
  KeyChip,
  PanelBody,
  PanelHeader,
  Spinner,
} from "@/components/ui";

export function CachePanel({ feed, threshold }: { feed: CacheFeed; threshold: number }) {
  const [query, setQuery] = useState("");
  const [expandedKey, setExpandedKey] = useState<string | null>(null);

  const needle = query.trim().toLowerCase();
  const visible = needle
    ? feed.entries.filter(
        (entry) =>
          entry.query.toLowerCase().includes(needle) ||
          entry.answer.toLowerCase().includes(needle),
      )
    : feed.entries;

  return (
    <section className="flex h-full min-h-0 flex-col bg-zinc-950">
      <PanelHeader
        icon={<DatabaseIcon className="size-4" />}
        title="Semantic Cache"
        subtitle={
          <span className="flex flex-wrap items-center gap-1.5">
            <KeyChip>idx:cache</KeyChip>
            <span>
              {feed.total} {feed.total === 1 ? "entry" : "entries"}
            </span>
            <span className="text-zinc-700">·</span>
            <span title="Cosine similarity a question must beat to count as a hit">
              hit ≥ {threshold}
            </span>
          </span>
        }
        actions={
          <>
            {feed.isRefreshing ? <Spinner className="size-3.5 text-zinc-600" /> : null}
            <IconButton label="Refresh cache list" onClick={feed.refresh} disabled={feed.isRefreshing}>
              <RefreshIcon className="size-4" />
            </IconButton>
          </>
        }
      />

      {feed.isStale ? (
        <p className="shrink-0 border-b border-amber-500/20 bg-amber-500/5 px-4 py-1.5 text-[11px] text-amber-400/90">
          Lost contact with Redis — showing the last known entries.
        </p>
      ) : null}

      {feed.entries.length > 0 ? (
        <div className="shrink-0 border-b border-zinc-800/80 p-2">
          <label className="relative block">
            <span className="sr-only">Filter cached questions</span>
            <SearchIcon className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-zinc-600" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Filter cached questions…"
              className="w-full rounded-md border border-zinc-800 bg-zinc-900/60 py-1.5 pr-2.5 pl-8 text-xs text-zinc-200 placeholder:text-zinc-600 focus:border-zinc-700 focus:ring-1 focus:ring-sky-500/40 focus:outline-none"
            />
          </label>
        </div>
      ) : null}

      <PanelBody>
        {feed.error ? (
          <ErrorState
            title="Cache unavailable"
            message={feed.error}
            hint={feed.hint}
            onRetry={feed.refresh}
          />
        ) : feed.entries.length === 0 ? (
          <EmptyState
            icon={<DatabaseIcon className="size-5" />}
            title="Nothing cached yet"
            description="Ask a question in the chat. Every answer generated from scratch gets written here, and the next similar question will match it."
          />
        ) : visible.length === 0 ? (
          <EmptyState
            icon={<SearchIcon className="size-5" />}
            title="No matches"
            description={`Nothing cached matches “${query.trim()}”.`}
          />
        ) : (
          <ul className="divide-y divide-zinc-800/60">
            {visible.map((entry) => (
              <CacheRow
                key={entry.key}
                entry={entry}
                expanded={expandedKey === entry.key}
                onToggle={() =>
                  setExpandedKey((current) => (current === entry.key ? null : entry.key))
                }
              />
            ))}
          </ul>
        )}
      </PanelBody>

      <footer className="shrink-0 border-t border-zinc-800/80 px-4 py-2 text-[11px] text-zinc-600">
        {feed.lastUpdated === null
          ? "Polling every 4s"
          : `Updated ${new Date(feed.lastUpdated).toLocaleTimeString([], { hour12: false })} · polling every 4s`}
      </footer>
    </section>
  );
}

function CacheRow({
  entry,
  expanded,
  onToggle,
}: {
  entry: CacheEntry;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <li>
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        className="w-full cursor-pointer px-4 py-3 text-left transition-colors hover:bg-zinc-900/60 focus-visible:bg-zinc-900/60 focus-visible:outline-none"
      >
        <p className="text-[13px] leading-snug font-medium text-zinc-200">{entry.query}</p>
        <p
          className={cx(
            "mt-1.5 text-xs leading-relaxed text-zinc-500",
            !expanded && "line-clamp-2",
          )}
        >
          {entry.answer}
        </p>
        <div className="mt-2 flex items-center gap-2">
          <KeyChip>{shortKey(entry.key)}</KeyChip>
          <span className="text-[11px] text-zinc-600">{formatAge(entry.ageSeconds)}</span>
          {expanded ? null : (
            <span className="ml-auto text-[11px] text-zinc-700">click to expand</span>
          )}
        </div>
        {expanded && entry.ttlSeconds !== null ? (
          <div className="mt-2">
            <Badge tone="neutral">expires in {formatCountdown(entry.ttlSeconds)}</Badge>
          </div>
        ) : null}
      </button>
    </li>
  );
}
