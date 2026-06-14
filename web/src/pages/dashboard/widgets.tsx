/**
 * Dashboard-local presentational widgets (PRD §4.1). These are page-private to the
 * Dashboard screen — they compose the shared design-system primitives (Panel,
 * Badge, …) into the small stat tiles and proportion bars the dashboard needs.
 * They never re-style the shared shell and own no data fetching.
 */

import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

/** A single big-number stat tile (totals row at the top of the dashboard). */
export function StatTile({
  label,
  value,
  tone,
  sub,
  dotClass,
}: {
  label: string;
  value: ReactNode;
  /** Tailwind text-color class for the big number (e.g. "text-active"). */
  tone?: string;
  sub?: ReactNode;
  /** Optional leading status dot color class. */
  dotClass?: string;
}) {
  return (
    <div className="flex flex-col gap-1 rounded-md border border-border bg-bg-raised px-3 py-2.5">
      <div className="flex items-center gap-1.5">
        {dotClass && <span className={cn("h-1.5 w-1.5 rounded-full", dotClass)} />}
        <span className="text-2xs font-medium uppercase tracking-wide text-text-faint">{label}</span>
      </div>
      <span className={cn("font-mono text-xl tabular-nums leading-none", tone ?? "text-text")}>{value}</span>
      {sub && <span className="text-2xs text-text-muted">{sub}</span>}
    </div>
  );
}

/**
 * A labeled proportion row: a left label, a count, and a fill bar showing this
 * row's share of `total`. Used by the type / tier / source breakdown panels.
 */
export function BreakdownRow({
  label,
  count,
  total,
  /** Tailwind bg color class for the fill (e.g. "bg-type-preference"). */
  fillClass = "bg-accent",
  /** Inline fill color (for runtime/free-form category colors); overrides fillClass. */
  fillColor,
  /** Optional rich label node (e.g. a <CategoryBadge/>) replacing the plain text label. */
  labelNode,
}: {
  label: string;
  count: number;
  total: number;
  fillClass?: string;
  fillColor?: string;
  labelNode?: ReactNode;
}) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  return (
    <div className="flex items-center gap-3">
      <div className="w-28 shrink-0 truncate">
        {labelNode ?? (
          <span className="font-mono text-xs text-text-muted" title={label}>
            {label}
          </span>
        )}
      </div>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-bg-inset">
        <div
          className={cn("h-full rounded-full transition-all duration-fast", !fillColor && fillClass)}
          style={{ width: `${pct}%`, ...(fillColor ? { backgroundColor: fillColor } : {}) }}
        />
      </div>
      <span className="w-10 shrink-0 text-right font-mono text-xs tabular-nums text-text">{count.toLocaleString()}</span>
      <span className="w-9 shrink-0 text-right font-mono text-2xs tabular-nums text-text-faint">
        {Math.round(pct)}%
      </span>
    </div>
  );
}

/** Empty-ish hint shown when a breakdown map has no entries. */
export function MutedHint({ children }: { children: ReactNode }) {
  return <div className="py-3 text-center text-2xs text-text-faint">{children}</div>;
}
