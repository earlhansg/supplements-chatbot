/** Shared presentational primitives used by all three panels. */

import { cx } from "@/lib/cx";
import { AlertIcon } from "@/components/icons";

/* ------------------------------------------------------------------ */
/* Panel shell                                                        */
/* ------------------------------------------------------------------ */

export function PanelHeader({
  icon,
  title,
  subtitle,
  actions,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <header className="flex shrink-0 items-start gap-3 border-b border-zinc-800/80 bg-zinc-900/40 px-4 py-3">
      <span className="mt-0.5 text-zinc-500">{icon}</span>
      <div className="min-w-0 flex-1">
        <h2 className="truncate text-sm font-semibold tracking-tight text-zinc-100">{title}</h2>
        {subtitle ? <div className="mt-0.5 text-xs text-zinc-500">{subtitle}</div> : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-1">{actions}</div> : null}
    </header>
  );
}

/** Vertically scrolling panel body with the thin custom scrollbar. */
export function PanelBody({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cx("panel-scroll min-h-0 flex-1 overflow-y-auto", className)}>{children}</div>
  );
}

/* ------------------------------------------------------------------ */
/* Badges                                                             */
/* ------------------------------------------------------------------ */

/**
 * Cache-hit green vs cache-miss amber is the single most important visual in
 * the app, so the two tones differ in hue, border and text weight — not just
 * shade — and each carries an explicit word rather than relying on colour alone.
 */
const BADGE_TONES = {
  hit: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  miss: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  error: "border-rose-500/40 bg-rose-500/10 text-rose-300",
  neutral: "border-zinc-700 bg-zinc-800/60 text-zinc-400",
  info: "border-sky-500/30 bg-sky-500/10 text-sky-300",
} as const;

export type BadgeTone = keyof typeof BADGE_TONES;

export function Badge({
  tone = "neutral",
  children,
  className,
}: {
  tone?: BadgeTone;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-medium leading-5 whitespace-nowrap",
        BADGE_TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/** Small monospace chip for Redis keys and index names. */
export function KeyChip({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <code
      className={cx(
        "rounded border border-zinc-800 bg-zinc-900 px-1.5 py-0.5 font-mono text-[11px] text-zinc-400",
        className,
      )}
    >
      {children}
    </code>
  );
}

/* ------------------------------------------------------------------ */
/* Status / feedback                                                  */
/* ------------------------------------------------------------------ */

export function Spinner({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={cx("animate-spin", className)}
      fill="none"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2.5" opacity="0.25" />
      <path
        d="M21 12a9 9 0 0 0-9-9"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function IconButton({
  label,
  onClick,
  disabled,
  children,
  className,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={label}
      aria-label={label}
      className={cx(
        "rounded-md p-1.5 text-zinc-500 transition-colors",
        "hover:bg-zinc-800 hover:text-zinc-200",
        "focus-visible:ring-2 focus-visible:ring-sky-500/60 focus-visible:outline-none",
        "disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-zinc-500",
        className,
      )}
    >
      {children}
    </button>
  );
}

export function EmptyState({
  icon,
  title,
  description,
  /** Centre within the full panel height. Turn off when stacking it above other
   *  content, otherwise it absorbs the free space and pushes siblings apart. */
  fill = true,
  width = "max-w-[26ch]",
}: {
  icon: React.ReactNode;
  title: string;
  description: React.ReactNode;
  fill?: boolean;
  width?: string;
}) {
  return (
    <div
      className={cx(
        "flex flex-col items-center justify-center gap-3 px-6 text-center",
        fill ? "h-full py-12" : "py-2",
      )}
    >
      <span className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-3 text-zinc-600">
        {icon}
      </span>
      <p className="text-sm font-medium text-zinc-400">{title}</p>
      <p className={cx("text-xs leading-relaxed text-zinc-600", width)}>{description}</p>
    </div>
  );
}

export function ErrorState({
  title,
  message,
  hint,
  onRetry,
}: {
  title: string;
  message: string;
  hint?: string | null;
  onRetry?: () => void;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 px-6 py-12 text-center">
      <span className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-rose-400">
        <AlertIcon className="size-5" />
      </span>
      <p className="text-sm font-medium text-rose-300">{title}</p>
      <p className="max-w-[34ch] text-xs leading-relaxed text-zinc-400">{message}</p>
      {hint ? (
        <p className="max-w-[34ch] text-xs leading-relaxed text-zinc-600">{hint}</p>
      ) : null}
      {onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          className="mt-1 rounded-md border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 transition-colors hover:border-zinc-600 hover:bg-zinc-800 focus-visible:ring-2 focus-visible:ring-sky-500/60 focus-visible:outline-none"
        >
          Try again
        </button>
      ) : null}
    </div>
  );
}

/** Loading placeholder rows, sized to roughly match real content. */
export function SkeletonList({ rows = 4 }: { rows?: number }) {
  return (
    <div className="space-y-2 p-3">
      {Array.from({ length: rows }, (_, index) => (
        <div key={index} className="animate-pulse rounded-lg border border-zinc-800/70 p-3">
          <div className="h-3 w-3/4 rounded bg-zinc-800" />
          <div className="mt-2 h-2.5 w-full rounded bg-zinc-800/60" />
          <div className="mt-1.5 h-2.5 w-5/6 rounded bg-zinc-800/60" />
        </div>
      ))}
    </div>
  );
}

/** Colour-coded connection dot used in the app header. */
export function StatusDot({ ok, label }: { ok: boolean | null; label: string }) {
  const tone =
    ok === null ? "bg-zinc-600" : ok ? "bg-emerald-400 shadow-emerald-400/50" : "bg-rose-500 shadow-rose-500/50";
  const text = ok === null ? "checking" : ok ? "connected" : "unreachable";

  return (
    <span
      className="inline-flex items-center gap-1.5 text-[11px] text-zinc-500"
      title={`${label}: ${text}`}
    >
      <span className={cx("size-1.5 rounded-full shadow-[0_0_6px_currentColor]", tone)} />
      <span className="hidden sm:inline">{label}</span>
    </span>
  );
}
