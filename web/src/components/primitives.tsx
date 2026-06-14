/**
 * Shared UI primitives (console chrome): Spinner, EmptyState, ErrorState, Panel,
 * Button, IconButton, Field/Label, Select, Input. These give every screen the same
 * dark, dense look. Screen agents should compose these rather than styling raw
 * HTML elements — that keeps spacing/borders/focus consistent.
 */

import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";
import { cn } from "@/lib/cn";

// --- Spinner ----------------------------------------------------------------

export function Spinner({ size = 16, className }: { size?: number; className?: string }) {
  return (
    <span
      role="status"
      aria-label="loading"
      className={cn("inline-block animate-spin rounded-full border-2 border-border-strong border-t-accent", className)}
      style={{ width: size, height: size }}
    />
  );
}

/** Centered loading state for a panel/page region. */
export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-12 text-sm text-text-muted">
      <Spinner /> {label}
    </div>
  );
}

// --- Empty / Error ----------------------------------------------------------

export function EmptyState({
  title = "Nothing here",
  hint,
  icon,
}: {
  title?: string;
  hint?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-12 text-center">
      {icon && <div className="text-text-faint">{icon}</div>}
      <div className="text-sm text-text-muted">{title}</div>
      {hint && <div className="max-w-md text-xs text-text-faint">{hint}</div>}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-12 text-center">
      <div className="font-mono text-sm text-danger">request failed</div>
      <div className="max-w-lg break-words text-xs text-text-muted">{message}</div>
      {onRetry && (
        <Button variant="ghost" onClick={onRetry}>
          retry
        </Button>
      )}
    </div>
  );
}

// --- Panel ------------------------------------------------------------------

export function Panel({
  title,
  actions,
  children,
  className,
  bodyClassName,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={cn("rounded-md border border-border bg-bg-raised", className)}>
      {(title || actions) && (
        <header className="flex h-9 items-center justify-between border-b border-border px-3">
          <h2 className="text-xs font-medium uppercase tracking-wide text-text-muted">{title}</h2>
          {actions && <div className="flex items-center gap-1.5">{actions}</div>}
        </header>
      )}
      <div className={cn("p-3", bodyClassName)}>{children}</div>
    </section>
  );
}

// --- Button -----------------------------------------------------------------

type ButtonVariant = "primary" | "default" | "ghost" | "danger";

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-accent text-text-inverse hover:bg-accent-hover border-transparent",
  default: "bg-bg-inset text-text hover:bg-bg-hover border-border-strong",
  ghost: "bg-transparent text-text-muted hover:text-text hover:bg-bg-hover border-transparent",
  danger: "bg-transparent text-danger hover:bg-danger/10 border-border-strong",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  loading?: boolean;
}

export function Button({ variant = "default", loading, children, className, disabled, ...rest }: ButtonProps) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={cn(
        "inline-flex items-center gap-1.5 rounded border px-2.5 py-1 text-xs font-medium transition-colors duration-fast disabled:cursor-not-allowed disabled:opacity-50",
        BUTTON_VARIANTS[variant],
        className,
      )}
    >
      {loading && <Spinner size={12} />}
      {children}
    </button>
  );
}

/** Square ghost icon button (toolbar actions). */
export function IconButton({ children, className, ...rest }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      className={cn(
        "inline-flex h-7 w-7 items-center justify-center rounded text-text-muted hover:bg-bg-hover hover:text-text",
        className,
      )}
    >
      {children}
    </button>
  );
}

// --- Form controls ----------------------------------------------------------

export function Input({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...rest}
      className={cn(
        "h-7 rounded border border-border-strong bg-bg px-2 text-xs text-text placeholder:text-text-faint focus:border-border-focus",
        className,
      )}
    />
  );
}

export function Select({ className, children, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...rest}
      className={cn(
        "h-7 rounded border border-border-strong bg-bg px-2 text-xs text-text focus:border-border-focus",
        className,
      )}
    >
      {children}
    </select>
  );
}

export function Field({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  return (
    <label className={cn("flex flex-col gap-1", className)}>
      <span className="text-2xs font-medium uppercase tracking-wide text-text-faint">{label}</span>
      {children}
    </label>
  );
}

/** A small key/value row used in detail panels. */
export function KeyValue({ k, children }: { k: string; children: ReactNode }) {
  return (
    <div className="flex items-start gap-3 py-1.5">
      <span className="w-28 shrink-0 text-2xs uppercase tracking-wide text-text-faint">{k}</span>
      <span className="min-w-0 flex-1 text-xs text-text">{children}</span>
    </div>
  );
}
