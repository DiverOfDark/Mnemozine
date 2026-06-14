/**
 * Logs screen — the filter toolbar (PRD §4.6).
 *
 * Renders the kind multi-toggle (ingest / extract / maintenance / injection) plus
 * source / session / ref-memory / since / until inputs. It is a controlled view over
 * useActivityFilters; the page owns the state. Kind chips use the per-kind token
 * color so the filter reads like the feed.
 */

import { ACTIVITY_KINDS, type ActivityKind } from "@/api";
import { Button, Field, Input } from "@/components";
import { cn } from "@/lib/cn";
import { ACTIVITY_KIND_LABEL, kindColorClass, kindDotClass } from "./activityMeta";
import type { ActivityFilterState } from "./useActivityFilters";

interface LogsToolbarProps {
  filters: ActivityFilterState;
  activeKinds: ActivityKind[];
  onToggleKind: (kind: ActivityKind) => void;
  onField: (field: keyof ActivityFilterState, value: string) => void;
  onReset: () => void;
  hasActiveFilters: boolean;
}

function KindToggle({
  kind,
  active,
  onClick,
}: {
  kind: ActivityKind;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      title={`filter: ${kind}`}
      className={cn(
        "inline-flex items-center gap-1.5 rounded border px-2 py-1 font-mono text-2xs uppercase tracking-wide transition-colors duration-fast",
        active
          ? cn("border-border-strong bg-bg-active", kindColorClass(kind))
          : "border-border bg-bg-inset text-text-faint hover:bg-bg-hover hover:text-text-muted",
      )}
    >
      <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", active ? kindDotClass(kind) : "bg-text-faint")} />
      {ACTIVITY_KIND_LABEL[kind]}
    </button>
  );
}

export function LogsToolbar({
  filters,
  activeKinds,
  onToggleKind,
  onField,
  onReset,
  hasActiveFilters,
}: LogsToolbarProps) {
  return (
    <div className="flex flex-col gap-3 border-b border-border bg-bg-raised px-5 py-3">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-2xs font-medium uppercase tracking-wide text-text-faint">kind</span>
        {ACTIVITY_KINDS.map((kind) => (
          <KindToggle
            key={kind}
            kind={kind}
            active={activeKinds.includes(kind)}
            onClick={() => onToggleKind(kind)}
          />
        ))}
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <Field label="source" className="w-36">
          <Input
            value={filters.source}
            placeholder="claude_code…"
            onChange={(e) => onField("source", e.target.value)}
          />
        </Field>
        <Field label="session" className="w-44">
          <Input
            value={filters.sessionId}
            placeholder="session id"
            className="font-mono"
            onChange={(e) => onField("sessionId", e.target.value)}
          />
        </Field>
        <Field label="ref memory" className="w-44">
          <Input
            value={filters.refMemoryId}
            placeholder="memory id"
            className="font-mono"
            onChange={(e) => onField("refMemoryId", e.target.value)}
          />
        </Field>
        <Field label="since" className="w-48">
          <Input
            type="datetime-local"
            value={filters.since}
            onChange={(e) => onField("since", e.target.value)}
          />
        </Field>
        <Field label="until" className="w-48">
          <Input
            type="datetime-local"
            value={filters.until}
            onChange={(e) => onField("until", e.target.value)}
          />
        </Field>
        <Button variant="ghost" onClick={onReset} disabled={!hasActiveFilters} className="mb-0.5">
          clear filters
        </Button>
      </div>
    </div>
  );
}
