/**
 * Recent activity feed (PRD §4.1). Page-local to the Dashboard. Reads the latest
 * ActivityEvents via useActivity (newest first) and renders a compact, kind-colored
 * feed. Each row links to the affected memory (first ref) and, for extract decisions,
 * surfaces the 4-way write decision. When the activity log is disabled the API
 * returns an empty feed by design — we say so honestly rather than implying silence.
 */

import { Link } from "react-router-dom";
import { Panel, Loading, ErrorState, EmptyState } from "@/components/primitives";
import type { ActivityEventOut, WriteDecision } from "@/api/types";
import { useActivity } from "@/api/hooks";
import { ACTIVITY_KIND_COLOR, WRITE_DECISION_COLOR } from "@/theme/tokens";
import { formatRelative, formatDateTime, shortId } from "@/lib/format";
import { PATHS } from "@/routes";
import { cn } from "@/lib/cn";

const KIND_LABEL: Record<string, string> = {
  ingest: "ingest",
  extract_decision: "extract",
  maintenance: "maint",
  injection: "inject",
};

/**
 * Kind → leading dot bg class. Written literally (not derived from
 * ACTIVITY_KIND_COLOR by string-munging) so Tailwind's JIT scanner always emits
 * these classes from this file, mirroring the text-color token map 1:1.
 */
const KIND_DOT: Record<string, string> = {
  ingest: "bg-info",
  extract_decision: "bg-type-preference",
  maintenance: "bg-warn",
  injection: "bg-tier-hot",
};

function decisionOf(event: ActivityEventOut): WriteDecision | null {
  const d = event.detail?.["decision"];
  return typeof d === "string" ? (d as WriteDecision) : null;
}

function FeedRow({ event }: { event: ActivityEventOut }) {
  const kindColor = ACTIVITY_KIND_COLOR[event.kind] ?? "text-text-muted";
  const dotColor = KIND_DOT[event.kind] ?? "bg-text-muted";
  const firstRef = event.ref_memory_ids[0];
  const decision = event.kind === "extract_decision" ? decisionOf(event) : null;

  return (
    <li className="flex items-start gap-2.5 py-1.5">
      <span className={cn("mt-1 h-1.5 w-1.5 shrink-0 rounded-full", dotColor)} aria-hidden />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <span className={cn("font-mono text-2xs uppercase tracking-wide", kindColor)}>
            {KIND_LABEL[event.kind] ?? event.kind}
          </span>
          {decision && (
            <span className={cn("font-mono text-2xs", WRITE_DECISION_COLOR[decision] ?? "text-text-faint")}>
              {decision}
            </span>
          )}
          {event.source && <span className="font-mono text-2xs text-text-faint">{event.source}</span>}
          <span className="ml-auto shrink-0 font-mono text-2xs text-text-faint" title={formatDateTime(event.ts)}>
            {formatRelative(event.ts)}
          </span>
        </div>
        <p className="truncate text-xs text-text" title={event.summary}>
          {event.summary}
        </p>
        {firstRef && (
          <Link
            to={PATHS.memoryDetail(firstRef)}
            className="font-mono text-2xs text-accent hover:underline"
            title={`open memory ${firstRef}`}
          >
            {shortId(firstRef, 12)}
            {event.ref_memory_ids.length > 1 && (
              <span className="ml-1 text-text-faint">+{event.ref_memory_ids.length - 1}</span>
            )}
          </Link>
        )}
      </div>
    </li>
  );
}

export function ActivityFeed({ scope, limit = 12 }: { scope: string | null; limit?: number }) {
  const project = scope && scope !== "global" ? scope.replace(/^project:/, "") : undefined;
  const { data, isLoading, error, refetch } = useActivity(
    { limit, ...(project ? { project } : {}) },
    { refetchInterval: 20_000 },
  );

  const items = data?.items ?? [];

  return (
    <Panel
      title="Recent activity"
      actions={
        <Link to={PATHS.logs} className="text-2xs text-accent hover:underline">
          view all →
        </Link>
      }
    >
      {isLoading ? (
        <Loading label="Loading activity…" />
      ) : error ? (
        <ErrorState error={error} onRetry={() => void refetch()} />
      ) : items.length === 0 ? (
        <EmptyState
          title="No recent activity"
          hint="The persisted activity log is off by default (MNEMOZINE_WEB__ENABLE_ACTIVITY_LOG=1). With it disabled this feed is empty by design."
        />
      ) : (
        <ul className="flex flex-col divide-y divide-border/50">
          {items.map((event) => (
            <FeedRow key={event.id} event={event} />
          ))}
        </ul>
      )}
    </Panel>
  );
}
