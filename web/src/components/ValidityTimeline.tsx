/**
 * ValidityTimeline — renders a memory's temporal validity window (the signature
 * feature, PRD §2/§4.3). An open window (valid_to=null, active) shows as an
 * ongoing bar to "now"; a closed window shows valid_from → valid_to and is
 * greyed. Use on the memory detail drawer and anywhere validity is surfaced.
 */

import { formatDateTime, formatRelative } from "@/lib/format";
import type { ValidityWindow } from "@/api/types";
import { cn } from "@/lib/cn";

interface ValidityTimelineProps {
  validity: ValidityWindow;
  className?: string;
}

export function ValidityTimeline({ validity, className }: ValidityTimelineProps) {
  const { valid_from, valid_to, active } = validity;
  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <div className="flex items-center gap-2">
        {/* start node */}
        <div className="flex flex-col items-center">
          <span className="h-2.5 w-2.5 rounded-full bg-accent" />
        </div>
        {/* track */}
        <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-bg-inset">
          <div
            className={cn("absolute inset-0 rounded-full", active ? "bg-active/70" : "bg-superseded/60")}
          />
        </div>
        {/* end node */}
        <div className="flex flex-col items-center">
          {active ? (
            <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-active" title="active (now)" />
          ) : (
            <span className="h-2.5 w-2.5 rounded-full bg-superseded" title="superseded" />
          )}
        </div>
      </div>
      <div className="flex items-center justify-between font-mono text-2xs text-text-muted">
        <span title={formatDateTime(valid_from)}>
          {formatDateTime(valid_from)}
          <span className="ml-1 text-text-faint">({formatRelative(valid_from)})</span>
        </span>
        {active ? (
          <span className="text-active">now · active</span>
        ) : (
          <span className="text-superseded" title={formatDateTime(valid_to)}>
            {formatDateTime(valid_to)} · closed
          </span>
        )}
      </div>
    </div>
  );
}
