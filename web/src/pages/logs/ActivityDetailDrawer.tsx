/**
 * Logs screen — the right-side drawer for a single ActivityEvent (PRD §4.6).
 *
 * Shows the event's metadata (kind / source / session / project / timestamp), the
 * 4-way write decision (for extract_decision events), the list of affected memories
 * as deep links to their detail route, and the raw structured `detail` via
 * <JsonViewer>. Pure presentation — the page owns open/close state.
 */

import { Link } from "react-router-dom";
import type { ActivityEventOut } from "@/api";
import { Badge, DetailDrawer, DrawerSection, JsonViewer, KeyValue } from "@/components";
import { formatDateTime, formatRelative } from "@/lib/format";
import { PATHS } from "@/routes";
import { cn } from "@/lib/cn";
import {
  ACTIVITY_KIND_LABEL,
  extractWriteDecision,
  kindColorClass,
  kindDotClass,
  writeDecisionColorClass,
} from "./activityMeta";

interface ActivityDetailDrawerProps {
  event: ActivityEventOut | null;
  onClose: () => void;
}

export function ActivityDetailDrawer({ event, onClose }: ActivityDetailDrawerProps) {
  if (!event) return null;

  const decision = extractWriteDecision(event);

  return (
    <DetailDrawer
      open={Boolean(event)}
      onClose={onClose}
      title={event.summary}
      subtitle={event.id}
      headerActions={
        <Badge
          textClass={kindColorClass(event.kind)}
          dotClass={kindDotClass(event.kind)}
          bgClass="bg-bg-inset"
        >
          {ACTIVITY_KIND_LABEL[event.kind]}
        </Badge>
      }
    >
      <DrawerSection title="event">
        <div className="rounded-md border border-border bg-bg-inset px-3 py-1">
          <KeyValue k="kind">
            <span className={cn("font-mono", kindColorClass(event.kind))}>{event.kind}</span>
          </KeyValue>
          {decision && (
            <KeyValue k="decision">
              <span className={cn("font-mono uppercase", writeDecisionColorClass(decision))}>{decision}</span>
            </KeyValue>
          )}
          <KeyValue k="source">
            <span className="font-mono">{event.source ?? "—"}</span>
          </KeyValue>
          <KeyValue k="session">
            <span className="break-all font-mono">{event.session_id ?? "—"}</span>
          </KeyValue>
          <KeyValue k="project">
            <span className="font-mono">{event.project ?? "—"}</span>
          </KeyValue>
          <KeyValue k="timestamp">
            <span className="font-mono">{formatDateTime(event.ts)}</span>
            <span className="ml-2 text-text-faint">({formatRelative(event.ts)})</span>
          </KeyValue>
        </div>
      </DrawerSection>

      <DrawerSection title={`affected memories (${event.ref_memory_ids.length})`}>
        {event.ref_memory_ids.length === 0 ? (
          <p className="text-2xs text-text-faint">No memories referenced by this event.</p>
        ) : (
          <ul className="flex flex-col gap-1.5">
            {event.ref_memory_ids.map((id) => (
              <li key={id}>
                <Link
                  to={PATHS.memoryDetail(id)}
                  className="inline-flex items-center gap-2 rounded border border-border bg-bg-inset px-2 py-1 font-mono text-2xs text-accent hover:border-border-strong hover:bg-bg-hover hover:text-accent-hover"
                >
                  <span className="text-text-faint">→</span>
                  {id}
                </Link>
              </li>
            ))}
          </ul>
        )}
      </DrawerSection>

      <DrawerSection title="raw detail">
        <JsonViewer value={event.detail} maxHeight={420} />
      </DrawerSection>
    </DetailDrawer>
  );
}
