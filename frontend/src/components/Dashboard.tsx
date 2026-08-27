"use client";

/**
 * The three-panel shell and the only place that owns shared state.
 *
 *   left   cached knowledge   polled from /api/cache
 *   middle chat               posts to /api/chat
 *   right  request log        derived from the chat responses, session-only
 *
 * The chat and the log are driven by the same request, so `sendQuestion` lives
 * here rather than inside `ChatPanel` — one fetch feeds both panels, and a cache
 * miss also nudges the left panel to refresh early instead of waiting out the
 * poll interval.
 */

import { useCallback, useEffect, useState } from "react";

import { cx } from "@/lib/cx";
import type { PublicConfig } from "@/lib/server-config";
import { useCacheEntries } from "@/lib/use-cache-entries";
import type {
  ApiError,
  CacheEntry,
  ChatMessage,
  ChatResponse,
  LogEntry,
} from "@/lib/types";
import { CachePanel } from "@/components/CachePanel";
import { ChatPanel } from "@/components/ChatPanel";
import { RequestLogPanel } from "@/components/RequestLogPanel";
import { ActivityIcon, CloseIcon, DatabaseIcon } from "@/components/icons";
import { StatusDot } from "@/components/ui";

const STATUS_POLL_INTERVAL_MS = 10_000;

type MobilePanel = "left" | "right" | null;

export function Dashboard({
  initialEntries,
  initialTotal,
  initialCacheError,
  config,
}: {
  initialEntries: CacheEntry[];
  initialTotal: number;
  initialCacheError: { error: string; hint?: string } | null;
  config: PublicConfig;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [pending, setPending] = useState(false);
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>(null);

  const cache = useCacheEntries(initialEntries, initialTotal, initialCacheError);
  const status = useUpstreamStatus();
  const refreshCache = cache.refresh;

  const sendQuestion = useCallback(
    async (question: string) => {
      setPending(true);
      setMessages((current) => [
        ...current,
        { id: newId(), role: "user", content: question },
      ]);

      // Client-side timing: the backend returns no duration of its own, so the
      // clock starts here and stops the moment the response body is parsed.
      const startedAt = performance.now();

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        });

        if (!response.ok) {
          const body = (await response.json().catch(() => null)) as ApiError | null;
          throw new Error(body?.error ?? `Request failed (${response.status})`);
        }

        const data = (await response.json()) as ChatResponse;
        const durationMs = performance.now() - startedAt;

        setMessages((current) => [
          ...current,
          {
            id: newId(),
            role: "assistant",
            content: data.answer,
            isCached: data.is_cached,
            similarity: data.cache_similarity,
            sources: data.sources ?? [],
            durationMs,
          },
        ]);

        setLog((current) => [
          {
            id: newId(),
            question,
            status: data.is_cached ? "hit" : "miss",
            durationMs,
            at: Date.now(),
            similarity: data.cache_similarity,
            sourceCount: data.sources?.length ?? 0,
          },
          ...current,
        ]);

        // A miss just wrote a new `cache:<uuid>` document — show it immediately.
        if (!data.is_cached) refreshCache();
      } catch (caught) {
        const durationMs = performance.now() - startedAt;
        const message =
          caught instanceof Error ? caught.message : "Something went wrong sending that question.";

        setMessages((current) => [
          ...current,
          { id: newId(), role: "error", content: message, durationMs },
        ]);

        setLog((current) => [
          {
            id: newId(),
            question,
            status: "error",
            durationMs,
            at: Date.now(),
            similarity: null,
            sourceCount: 0,
            error: message,
          },
          ...current,
        ]);
      } finally {
        setPending(false);
      }
    },
    [refreshCache],
  );

  // Escape closes whichever drawer is open on small screens.
  useEffect(() => {
    if (mobilePanel === null) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobilePanel(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [mobilePanel]);

  return (
    <div className="flex h-dvh flex-col overflow-hidden">
      <AppHeader
        status={status}
        cacheCount={cache.total}
        logCount={log.length}
        mobilePanel={mobilePanel}
        onToggle={(panel) => setMobilePanel((current) => (current === panel ? null : panel))}
      />

      <div className="relative flex min-h-0 flex-1">
        {/* Backdrop for the mobile drawers. */}
        {mobilePanel !== null ? (
          <button
            type="button"
            aria-label="Close panel"
            onClick={() => setMobilePanel(null)}
            className="absolute inset-0 z-30 bg-black/60 backdrop-blur-[2px] lg:hidden"
          />
        ) : null}

        <Drawer side="left" open={mobilePanel === "left"} onClose={() => setMobilePanel(null)}>
          <CachePanel feed={cache} threshold={config.cacheSimilarityThreshold} />
        </Drawer>

        <main className="flex min-w-0 flex-1 flex-col border-zinc-800/80 lg:border-x">
          <ChatPanel
            messages={messages}
            pending={pending}
            onSend={sendQuestion}
            backendOnline={status.backend}
          />
        </main>

        <Drawer side="right" open={mobilePanel === "right"} onClose={() => setMobilePanel(null)}>
          <RequestLogPanel entries={log} onClear={() => setLog([])} />
        </Drawer>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Layout pieces                                                      */
/* ------------------------------------------------------------------ */

/**
 * A side panel: a fixed column at `lg` and up, an overlay drawer below it.
 * Both variants render the same children, so panel state survives toggling.
 */
function Drawer({
  side,
  open,
  onClose,
  children,
}: {
  side: "left" | "right";
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const hiddenTransform = side === "left" ? "-translate-x-full" : "translate-x-full";

  return (
    <aside
      className={cx(
        // Opaque: as an overlay it sits on top of the chat, which would
        // otherwise show through the panel's own translucent surfaces.
        "absolute inset-y-0 z-40 flex w-[85vw] max-w-sm flex-col bg-zinc-950 shadow-2xl shadow-black/60 transition-transform duration-200 ease-out",
        "lg:static lg:z-auto lg:w-80 lg:max-w-none lg:translate-x-0 lg:shadow-none xl:w-96",
        side === "left" ? "left-0" : "right-0",
        open ? "translate-x-0" : hiddenTransform,
      )}
    >
      {/* A dedicated strip rather than an overlay button — floating it on top of
          the panel header collided with that panel's own action buttons. */}
      <div className="flex shrink-0 items-center justify-end border-b border-zinc-800 bg-zinc-900 px-2 py-1 lg:hidden">
        <button
          type="button"
          onClick={onClose}
          aria-label="Close panel"
          className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
        >
          <CloseIcon className="size-4" />
        </button>
      </div>
      {children}
    </aside>
  );
}

function AppHeader({
  status,
  cacheCount,
  logCount,
  mobilePanel,
  onToggle,
}: {
  status: UpstreamStatus;
  cacheCount: number;
  logCount: number;
  mobilePanel: MobilePanel;
  onToggle: (panel: "left" | "right") => void;
}) {
  return (
    <header className="flex shrink-0 items-center gap-3 border-b border-zinc-800 bg-zinc-900/60 px-3 py-2.5 backdrop-blur">
      <MobileToggle
        label="Cached knowledge"
        count={cacheCount}
        active={mobilePanel === "left"}
        onClick={() => onToggle("left")}
      >
        <DatabaseIcon className="size-4" />
      </MobileToggle>

      <div className="flex min-w-0 flex-1 items-center gap-2.5">
        <span className="hidden size-7 shrink-0 place-items-center rounded-lg bg-gradient-to-br from-sky-500 to-emerald-500 text-[13px] font-bold text-white sm:grid">
          S
        </span>
        <div className="min-w-0">
          <h1 className="truncate text-sm font-semibold tracking-tight text-zinc-100">
            Supplements Store Chatbot
          </h1>
          <p className="truncate text-[11px] text-zinc-500">
            Semantic cache demo · FastAPI + LangGraph + Redis
          </p>
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-3">
        <StatusDot ok={status.backend} label="API" />
        <StatusDot ok={status.redis} label="Redis" />
      </div>

      <MobileToggle
        label="Request log"
        count={logCount}
        active={mobilePanel === "right"}
        onClick={() => onToggle("right")}
      >
        <ActivityIcon className="size-4" />
      </MobileToggle>
    </header>
  );
}

function MobileToggle({
  label,
  count,
  active,
  onClick,
  children,
}: {
  label: string;
  count: number;
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      aria-pressed={active}
      className={cx(
        "relative shrink-0 rounded-md border p-1.5 transition-colors lg:hidden",
        active
          ? "border-zinc-700 bg-zinc-800 text-zinc-100"
          : "border-zinc-800 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300",
      )}
    >
      {children}
      {count > 0 ? (
        <span className="absolute -top-1 -right-1 grid min-w-4 place-items-center rounded-full bg-sky-600 px-1 text-[10px] leading-4 font-semibold text-white">
          {count > 99 ? "99+" : count}
        </span>
      ) : null}
    </button>
  );
}

/* ------------------------------------------------------------------ */

interface UpstreamStatus {
  backend: boolean | null;
  redis: boolean | null;
}

/** Polls `/api/status` so an unreachable backend is obvious before you send. */
function useUpstreamStatus(): UpstreamStatus {
  const [status, setStatus] = useState<UpstreamStatus>({ backend: null, redis: null });

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      try {
        const response = await fetch("/api/status", { cache: "no-store" });
        if (!response.ok) throw new Error();
        const data = (await response.json()) as UpstreamStatus;
        if (!cancelled) setStatus(data);
      } catch {
        if (!cancelled) setStatus({ backend: false, redis: false });
      }
    };

    void check();
    const id = setInterval(() => void check(), STATUS_POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return status;
}

function newId(): string {
  return crypto.randomUUID();
}
