"use client";

/**
 * Middle panel — the chat itself. Talks to `/api/chat`, which proxies to the
 * FastAPI `/chat` endpoint. No mock data anywhere in this file.
 */

import { useEffect, useRef, useState } from "react";

import { SUGGESTED_QUESTIONS } from "@/lib/constants";
import { cx } from "@/lib/cx";
import { formatDuration, formatSimilarity } from "@/lib/format";
import type { AssistantMessage, ChatMessage, ErrorMessage } from "@/lib/types";
import { AlertIcon, BoltIcon, ChatIcon, ChevronIcon, SendIcon } from "@/components/icons";
import { Badge, EmptyState, PanelBody, PanelHeader } from "@/components/ui";

export function ChatPanel({
  messages,
  pending,
  onSend,
  backendOnline,
}: {
  messages: ChatMessage[];
  pending: boolean;
  onSend: (question: string) => void;
  backendOnline: boolean | null;
}) {
  const [draft, setDraft] = useState("");
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, pending]);

  const submit = (text: string) => {
    const question = text.trim();
    if (!question || pending) return;
    onSend(question);
    setDraft("");
    inputRef.current?.focus();
  };

  return (
    <section className="flex h-full min-h-0 flex-col bg-zinc-950/40">
      <PanelHeader
        icon={<ChatIcon className="size-4" />}
        title="Support Chat"
        subtitle={
          backendOnline === false
            ? "Backend unreachable — messages will fail until uvicorn is running"
            : "Answers come from the FastAPI + LangGraph backend"
        }
      />

      <PanelBody className="px-4">
        {messages.length === 0 && !pending ? (
          <div className="flex h-full flex-col items-center justify-center gap-7 py-10">
            <EmptyState
              icon={<ChatIcon className="size-5" />}
              title="Ask the store anything"
              description="The backend checks its semantic cache first. A miss retrieves FAQs and calls the LLM, then caches the result."
              fill={false}
              width="max-w-[46ch]"
            />
            <div className="w-full max-w-md">
              <p className="mb-2 text-center text-[11px] tracking-wide text-zinc-600 uppercase">
                Try one of these
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTED_QUESTIONS.map((question) => (
                  <button
                    key={question}
                    type="button"
                    onClick={() => submit(question)}
                    className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1.5 text-xs text-zinc-400 transition-colors hover:border-zinc-700 hover:bg-zinc-800 hover:text-zinc-200 focus-visible:ring-2 focus-visible:ring-sky-500/60 focus-visible:outline-none"
                  >
                    {question}
                  </button>
                ))}
              </div>
              <p className="mt-3 text-center text-[11px] leading-relaxed text-zinc-700">
                Each pair is a question and a paraphrase of it. Ask one, then the
                other — the second comes back as a cache hit.
              </p>
            </div>
          </div>
        ) : (
          <div className="mx-auto flex max-w-3xl flex-col gap-5 py-5">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            {pending ? <PendingBubble /> : null}
            <div ref={bottomRef} />
          </div>
        )}
      </PanelBody>

      <footer className="shrink-0 border-t border-zinc-800/80 bg-zinc-900/30 p-3">
        <form
          className="mx-auto flex max-w-3xl items-end gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            submit(draft);
          }}
        >
          <textarea
            ref={inputRef}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              // Enter sends; Shift+Enter inserts a newline.
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit(draft);
              }
            }}
            rows={1}
            disabled={pending}
            placeholder={pending ? "Waiting for the backend…" : "Ask about shipping, refunds, products…"}
            className="panel-scroll max-h-32 min-h-[42px] flex-1 resize-none rounded-lg border border-zinc-800 bg-zinc-900/70 px-3 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-zinc-700 focus:ring-1 focus:ring-sky-500/40 focus:outline-none disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={pending || draft.trim().length === 0}
            className="inline-flex h-[42px] shrink-0 items-center gap-2 rounded-lg bg-sky-600 px-4 text-sm font-medium text-white transition-colors hover:bg-sky-500 focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 focus-visible:ring-offset-zinc-950 focus-visible:outline-none disabled:cursor-not-allowed disabled:bg-zinc-800 disabled:text-zinc-500"
          >
            <SendIcon className="size-4" />
            <span className="hidden sm:inline">Send</span>
          </button>
        </form>
        <p className="mx-auto mt-2 max-w-3xl text-[11px] text-zinc-700">
          Enter to send · Shift+Enter for a new line
        </p>
      </footer>
    </section>
  );
}

/* ------------------------------------------------------------------ */

function MessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-sky-600/90 px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap text-white">
          {message.content}
        </div>
      </div>
    );
  }

  if (message.role === "error") {
    return <ErrorBubble message={message} />;
  }

  return <AssistantBubble message={message} />;
}

function AssistantBubble({ message }: { message: AssistantMessage }) {
  const [showSources, setShowSources] = useState(false);

  return (
    <div className="flex flex-col items-start gap-1.5">
      <div className="max-w-[85%] rounded-2xl rounded-bl-md border border-zinc-800 bg-zinc-900/80 px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap text-zinc-200">
        {message.content}
      </div>

      <div className="flex flex-wrap items-center gap-2 pl-1">
        {message.isCached ? (
          <Badge tone="hit">
            <BoltIcon className="size-3" />
            Cache Hit
          </Badge>
        ) : (
          <Badge tone="miss">Cache Miss</Badge>
        )}

        <span className="font-mono text-[11px] text-zinc-500">
          {formatDuration(message.durationMs)}
        </span>

        {message.similarity !== null ? (
          <span
            className="text-[11px] text-zinc-600"
            title="Cosine similarity between this question and the cached one"
          >
            {formatSimilarity(message.similarity)} match
          </span>
        ) : null}

        {message.sources.length > 0 ? (
          <button
            type="button"
            onClick={() => setShowSources((value) => !value)}
            aria-expanded={showSources}
            className="inline-flex items-center gap-0.5 text-[11px] text-zinc-600 transition-colors hover:text-zinc-400"
          >
            {message.sources.length} FAQ{message.sources.length === 1 ? "" : "s"} retrieved
            <ChevronIcon
              className={cx("size-3 transition-transform", showSources && "rotate-180")}
            />
          </button>
        ) : null}
      </div>

      {showSources ? (
        <ul className="ml-1 max-w-[85%] space-y-1 rounded-lg border border-zinc-800/80 bg-zinc-900/40 p-2.5">
          {message.sources.map((source) => (
            <li key={source.id} className="flex items-start gap-2 text-[11px]">
              <code className="mt-px shrink-0 font-mono text-zinc-600">{source.id}</code>
              <span className="min-w-0 flex-1 text-zinc-500">{source.question}</span>
              <span className="shrink-0 text-zinc-700">{formatSimilarity(source.similarity)}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function ErrorBubble({ message }: { message: ErrorMessage }) {
  return (
    <div className="flex flex-col items-start gap-1.5">
      <div className="flex max-w-[85%] items-start gap-2.5 rounded-2xl rounded-bl-md border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm leading-relaxed text-rose-200">
        <AlertIcon className="mt-0.5 size-4 shrink-0 text-rose-400" />
        <span>{message.content}</span>
      </div>
      <div className="flex items-center gap-2 pl-1">
        <Badge tone="error">Failed</Badge>
        <span className="font-mono text-[11px] text-zinc-500">
          {formatDuration(message.durationMs)}
        </span>
      </div>
    </div>
  );
}

/**
 * While a request is in flight the elapsed time ticks up. On a cache hit it
 * stops in the tens of milliseconds; on a miss it visibly climbs — which is
 * exactly the contrast the demo is meant to show.
 */
function PendingBubble() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const startedAt = performance.now();
    const id = setInterval(() => setElapsed(performance.now() - startedAt), 60);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex flex-col items-start gap-1.5">
      <div className="flex items-center gap-2.5 rounded-2xl rounded-bl-md border border-zinc-800 bg-zinc-900/80 px-4 py-3">
        <span className="flex gap-1" aria-hidden="true">
          {[0, 1, 2].map((index) => (
            <span
              key={index}
              className="size-1.5 animate-bounce rounded-full bg-zinc-500"
              style={{ animationDelay: `${index * 120}ms` }}
            />
          ))}
        </span>
        <span className="text-sm text-zinc-500">Checking the semantic cache…</span>
      </div>
      <span className="pl-1 font-mono text-[11px] text-zinc-600 tabular-nums">
        {formatDuration(elapsed)}
      </span>
    </div>
  );
}
