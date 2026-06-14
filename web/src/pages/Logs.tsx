/**
 * Activity / Logs (PRD §4.6, FE-C) — a chronological, filterable ActivityEvent feed:
 * ingestion · extraction (the 4-way write decision) · maintenance · injection. Each
 * row links to the memories it affected and expands to its structured detail.
 *
 * Wires useActivity (kind is a repeatable filter passed as ActivityKind[]); folds the
 * top-bar scope (useScope) into the `project` filter. The feed is empty unless the
 * backend activity log is enabled (MNEMOZINE_WEB__ENABLE_ACTIVITY_LOG=1) — surfaced
 * as a hint in the empty state. This file owns only itself + src/pages/logs/**.
 */

import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import type { ActivityEventOut } from "@/api";
import { useActivity, useHealth } from "@/api";
import {
  Badge,
  type Column,
  DataTable,
  KeyboardHints,
  PageHeader,
} from "@/components";
import { useScope } from "@/state/scope";
import { formatDateTime, formatRelative } from "@/lib/format";
import { PATHS } from "@/routes";
import { cn } from "@/lib/cn";
import { ActivityDetailDrawer } from "./logs/ActivityDetailDrawer";
import { LogsToolbar } from "./logs/LogsToolbar";
import {
  ACTIVITY_KIND_LABEL,
  extractWriteDecision,
  kindColorClass,
  kindDotClass,
  writeDecisionColorClass,
} from "./logs/activityMeta";
import { ACTIVITY_PAGE_SIZE, useActivityFilters } from "./logs/useActivityFilters";

export default function Logs() {
  const { scope } = useScope();
  const {
    filters,
    query,
    offset,
    toggleKind,
    setField,
    setOffset,
    reset,
    hasActiveFilters,
  } = useActivityFilters(scope);

  const activity = useActivity(query, { placeholderData: (prev) => prev });
  const health = useHealth();
  const [selected, setSelected] = useState<ActivityEventOut | null>(null);

  const events = activity.data?.items ?? [];
  const page = activity.data?.page;
  const total = page?.total ?? 0;
  const activityEnabled = health.data?.activity_log_enabled;

  const columns = useMemo<Column<ActivityEventOut>[]>(
    () => [
      {
        id: "ts",
        header: "when",
        width: 150,
        className: "whitespace-nowrap",
        cell: (row) => (
          <span className="font-mono text-2xs text-text-muted" title={formatDateTime(row.ts)}>
            {formatRelative(row.ts)}
          </span>
        ),
      },
      {
        id: "kind",
        header: "kind",
        width: 132,
        cell: (row) => {
          const decision = extractWriteDecision(row);
          return (
            <span className="inline-flex items-center gap-2">
              <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", kindDotClass(row.kind))} />
              <span className={cn("font-mono text-2xs uppercase", kindColorClass(row.kind))}>
                {ACTIVITY_KIND_LABEL[row.kind]}
              </span>
              {decision && (
                <span
                  className={cn("font-mono text-2xs uppercase", writeDecisionColorClass(decision))}
                  title={`write decision: ${decision}`}
                >
                  {decision}
                </span>
              )}
            </span>
          );
        },
      },
      {
        id: "summary",
        header: "summary",
        cell: (row) => <span className="text-text">{row.summary}</span>,
      },
      {
        id: "source",
        header: "source",
        width: 120,
        cell: (row) => (
          <span className="font-mono text-2xs text-text-muted">{row.source ?? "—"}</span>
        ),
      },
      {
        id: "refs",
        header: "memories",
        width: 160,
        cell: (row) => {
          if (row.ref_memory_ids.length === 0) return <span className="text-text-faint">—</span>;
          const [first, ...rest] = row.ref_memory_ids;
          return (
            <span className="inline-flex items-center gap-1.5">
              <Link
                to={PATHS.memoryDetail(first!)}
                onClick={(e) => e.stopPropagation()}
                className="font-mono text-2xs text-accent hover:text-accent-hover hover:underline"
                title={first}
              >
                {first!.slice(0, 8)}…
              </Link>
              {rest.length > 0 && (
                <Badge bgClass="bg-bg-inset" title={row.ref_memory_ids.join(", ")}>
                  +{rest.length}
                </Badge>
              )}
            </span>
          );
        },
      },
    ],
    [],
  );

  const emptyHint = activityEnabled === false ? (
    <>
      The persisted activity log is <span className="text-warn">disabled</span> on this backend. Set{" "}
      <code className="font-mono text-text-muted">MNEMOZINE_WEB__ENABLE_ACTIVITY_LOG=1</code> and restart
      to record ingest / extract / maintenance / injection events.
    </>
  ) : hasActiveFilters ? (
    "No events match the current filters."
  ) : (
    "No activity recorded yet."
  );

  const rangeStart = total === 0 ? 0 : offset + 1;
  const rangeEnd = Math.min(offset + ACTIVITY_PAGE_SIZE, total);
  const canPrev = offset > 0;
  const canNext = offset + ACTIVITY_PAGE_SIZE < total;

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        title="Activity / Logs"
        subtitle={
          activityEnabled === false
            ? "activity log disabled — feed is empty by design"
            : `${total.toLocaleString()} events`
        }
        actions={
          <KeyboardHints
            hints={[
              { keys: ["j", "k"], label: "navigate" },
              { keys: ["↵"], label: "open detail" },
            ]}
          />
        }
      />

      <LogsToolbar
        filters={filters}
        activeKinds={filters.kinds}
        onToggleKind={toggleKind}
        onField={setField}
        onReset={reset}
        hasActiveFilters={hasActiveFilters}
      />

      <div className="min-h-0 flex-1 overflow-hidden">
        <DataTable
          columns={columns}
          rows={events}
          rowKey={(row) => row.id}
          onRowActivate={(row) => setSelected(row)}
          selectedKey={selected?.id ?? null}
          isLoading={activity.isLoading}
          error={activity.error}
          onRetry={() => void activity.refetch()}
          emptyTitle="No activity"
          emptyHint={emptyHint}
          className="h-full"
        />
      </div>

      <div className="flex shrink-0 items-center justify-between border-t border-border bg-bg-raised px-5 py-2">
        <span className="font-mono text-2xs text-text-faint">
          {total === 0 ? "0 events" : `${rangeStart}–${rangeEnd} of ${total.toLocaleString()}`}
        </span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setOffset(Math.max(0, offset - ACTIVITY_PAGE_SIZE))}
            disabled={!canPrev}
            className="rounded border border-border-strong bg-bg-inset px-2 py-1 text-2xs text-text-muted hover:bg-bg-hover hover:text-text disabled:cursor-not-allowed disabled:opacity-40"
          >
            ← prev
          </button>
          <button
            type="button"
            onClick={() => setOffset(offset + ACTIVITY_PAGE_SIZE)}
            disabled={!canNext}
            className="rounded border border-border-strong bg-bg-inset px-2 py-1 text-2xs text-text-muted hover:bg-bg-hover hover:text-text disabled:cursor-not-allowed disabled:opacity-40"
          >
            next →
          </button>
        </div>
      </div>

      <ActivityDetailDrawer event={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
