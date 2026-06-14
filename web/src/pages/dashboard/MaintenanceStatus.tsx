/**
 * Maintenance job status (PRD §4.1). Page-local to the Dashboard. Reads
 * useMaintenance and renders the scheduler state + a compact per-job grid
 * (enabled, last run, next run). Deep controls (trigger, merge review) live on the
 * Ops screen — this is the at-a-glance dashboard view that links there.
 */

import { Link } from "react-router-dom";
import { Panel, Loading, ErrorState, EmptyState } from "@/components/primitives";
import type { MaintenanceJobStatus } from "@/api/types";
import { useMaintenance } from "@/api/hooks";
import { formatRelative, formatDateTime } from "@/lib/format";
import { PATHS } from "@/routes";
import { cn } from "@/lib/cn";

function JobTile({ job }: { job: MaintenanceJobStatus }) {
  return (
    <div className="flex flex-col gap-1 rounded border border-border bg-bg-inset px-2.5 py-2">
      <div className="flex items-center gap-1.5">
        <span
          className={cn("h-1.5 w-1.5 rounded-full", job.enabled ? "bg-ok" : "bg-text-faint")}
          title={job.enabled ? "enabled" : "disabled"}
        />
        <span className="truncate font-mono text-xs text-text" title={job.name}>
          {job.name}
        </span>
      </div>
      <div className="flex items-center justify-between text-2xs text-text-faint">
        <span title={formatDateTime(job.last_run)}>last {formatRelative(job.last_run)}</span>
        {job.next_run && <span title={formatDateTime(job.next_run)}>next {formatRelative(job.next_run)}</span>}
      </div>
      {job.last_report && job.last_report.notes.length > 0 && (
        <p className="truncate text-2xs text-text-muted" title={job.last_report.notes.join(" · ")}>
          {job.last_report.notes[0]}
        </p>
      )}
    </div>
  );
}

export function MaintenanceStatus() {
  const { data, isLoading, error, refetch } = useMaintenance({ refetchInterval: 30_000 });
  const jobs = data?.jobs ?? [];

  return (
    <Panel
      title="Maintenance"
      actions={
        <div className="flex items-center gap-2">
          {data && (
            <span className="flex items-center gap-1.5 text-2xs text-text-muted" title={`cron: ${data.cron}`}>
              <span
                className={cn(
                  "h-1.5 w-1.5 rounded-full",
                  data.scheduler_running ? "animate-pulse bg-ok" : "bg-text-faint",
                )}
              />
              {data.scheduler_running ? "scheduler running" : "scheduler idle"}
            </span>
          )}
          <Link to={PATHS.maintenance} className="text-2xs text-accent hover:underline">
            ops →
          </Link>
        </div>
      }
    >
      {isLoading ? (
        <Loading label="Loading maintenance…" />
      ) : error ? (
        <ErrorState error={error} onRetry={() => void refetch()} />
      ) : jobs.length === 0 ? (
        <EmptyState title="No maintenance jobs reported" />
      ) : (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {jobs.map((job) => (
            <JobTile key={job.name} job={job} />
          ))}
        </div>
      )}
    </Panel>
  );
}
