/**
 * SchedulerPanel — Maintenance / Ops scheduler status (PRD §4.7).
 *
 * Shows whether the scheduler loop is running, the cron expression, and a per-job
 * table: enabled state, last run (relative), next run, and the last report counts.
 * Each row carries an inline trigger-job button wired to useRunMaintenance, and a
 * dedicated "trigger on demand" grid exposes every modeled job (consolidate, decay,
 * entity-resolution, audit, migrate-index) even when the scheduler omits it. The
 * full last-run report is inspectable per job via the JsonViewer.
 *
 * Page-local: consumes only shared design-system components (from "@/components")
 * + the typed api hooks/types (from "@/api"). It edits no shared files.
 */

import { useState } from "react";

import {
  useRunMaintenance,
  type MaintenanceJobName,
  type MaintenanceJobStatus,
  type MaintenanceReportOut,
  type MaintenanceStatusResponse,
} from "@/api";
import {
  Badge,
  Button,
  DataTable,
  ErrorState,
  JsonViewer,
  Loading,
  Panel,
  type Column,
} from "@/components";
import { formatDateTime, formatRelative } from "@/lib/format";
import { JOB_META, canonicalJobName, jobLabel } from "@/pages/maintenance/jobMeta";

interface SchedulerPanelProps {
  status: MaintenanceStatusResponse | undefined;
  isLoading: boolean;
  error: unknown;
  onRetry: () => void;
}

export function SchedulerPanel({ status, isLoading, error, onRetry }: SchedulerPanelProps) {
  const run = useRunMaintenance();
  const [openReport, setOpenReport] = useState<{ job: string; report: MaintenanceReportOut } | null>(null);

  if (isLoading && !status) return <Loading label="Loading scheduler status…" />;
  if (error && !status) return <ErrorState error={error} onRetry={onRetry} />;
  if (!status) return null;

  const jobs = status.jobs ?? [];

  const columns: Column<MaintenanceJobStatus>[] = [
    {
      id: "job",
      header: "Job",
      cell: (row) => (
        <div className="flex flex-col">
          <span className="font-medium text-text">{jobLabel(row.name)}</span>
          <span className="font-mono text-2xs text-text-faint">{row.name}</span>
        </div>
      ),
    },
    {
      id: "enabled",
      header: "Enabled",
      width: 96,
      cell: (row) =>
        row.enabled ? (
          <Badge textClass="text-ok" bgClass="bg-tier-bg-hot" dotClass="bg-ok">
            enabled
          </Badge>
        ) : (
          <Badge textClass="text-text-faint" bgClass="bg-bg-inset" dotClass="bg-text-faint">
            disabled
          </Badge>
        ),
    },
    {
      id: "last_run",
      header: "Last run",
      width: 132,
      cell: (row) => (
        <span className="text-text-muted" title={formatDateTime(row.last_run)}>
          {formatRelative(row.last_run)}
        </span>
      ),
    },
    {
      id: "next_run",
      header: "Next run",
      width: 148,
      cell: (row) => <span className="text-text-muted">{formatDateTime(row.next_run)}</span>,
    },
    {
      id: "report",
      header: "Last report",
      cell: (row) => (
        <ReportSummary
          report={row.last_report}
          onOpen={() => row.last_report && setOpenReport({ job: row.name, report: row.last_report })}
        />
      ),
    },
    {
      id: "trigger",
      header: "",
      width: 84,
      align: "right",
      cell: (row) => {
        const canonical = canonicalJobName(row.name);
        return (
          <Button
            variant="default"
            disabled={!canonical}
            loading={run.isPending && run.variables === canonical}
            onClick={(e) => {
              e.stopPropagation();
              if (canonical) run.mutate(canonical);
            }}
          >
            run
          </Button>
        );
      },
    },
  ];

  return (
    <div className="flex flex-col gap-3">
      <Panel
        title="Scheduler"
        actions={<SchedulerStateChip running={status.scheduler_running} cron={status.cron} />}
        bodyClassName="p-0"
      >
        <DataTable
          columns={columns}
          rows={jobs}
          rowKey={(r) => r.name}
          keyboardNav={false}
          emptyTitle="No scheduled jobs"
          emptyHint="The scheduler reports no jobs. Trigger one on demand from the panel below."
        />
        {run.isError && (
          <div className="border-t border-border px-3 py-2 text-2xs text-danger">
            trigger failed: {run.error instanceof Error ? run.error.message : String(run.error)}
          </div>
        )}
      </Panel>

      <TriggerGrid run={run} />

      {openReport && (
        <Panel
          title={`Last report — ${jobLabel(openReport.job)}`}
          actions={
            <Button variant="ghost" onClick={() => setOpenReport(null)}>
              close
            </Button>
          }
        >
          <JsonViewer value={openReport.report} maxHeight={280} />
        </Panel>
      )}
    </div>
  );
}

function SchedulerStateChip({ running, cron }: { running: boolean; cron: string }) {
  return (
    <div className="flex items-center gap-2">
      <Badge
        textClass={running ? "text-ok" : "text-text-faint"}
        bgClass={running ? "bg-tier-bg-hot" : "bg-bg-inset"}
        dotClass={running ? "bg-ok" : "bg-text-faint"}
      >
        {running ? "running" : "stopped"}
      </Badge>
      <code className="font-mono text-2xs text-text-muted" title="cron schedule">
        {cron || "—"}
      </code>
    </div>
  );
}

function ReportSummary({ report, onOpen }: { report: MaintenanceReportOut | null; onOpen: () => void }) {
  if (!report) return <span className="text-text-faint">—</span>;
  const counts: Array<[string, number]> = [
    ["consolidated", report.consolidated],
    ["merged", report.entities_merged],
    ["archived", report.archived],
    ["pruned", report.edges_pruned],
  ];
  const nonZero = counts.filter(([, v]) => v > 0);
  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        onOpen();
      }}
      className="flex flex-wrap items-center gap-1.5 text-left"
    >
      {nonZero.length === 0 ? (
        <span className="text-2xs text-text-faint">no changes</span>
      ) : (
        nonZero.map(([k, v]) => (
          <Badge key={k} textClass="text-text-muted" bgClass="bg-bg-inset">
            {k} {v}
          </Badge>
        ))
      )}
      {report.notes.length > 0 && (
        <span className="text-2xs text-text-faint" title={report.notes.join("\n")}>
          · {report.notes.length} note{report.notes.length === 1 ? "" : "s"}
        </span>
      )}
    </button>
  );
}

/**
 * TriggerGrid — explicit on-demand trigger buttons for every modeled job, so the
 * operator can run a job even when the scheduler reports it as not configured.
 */
function TriggerGrid({ run }: { run: ReturnType<typeof useRunMaintenance> }) {
  const lastRun = run.isSuccess ? run.data : null;
  return (
    <Panel title="Trigger job on demand">
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {JOB_META.map((job) => (
          <TriggerCard
            key={job.name}
            job={job.name}
            label={job.label}
            description={job.description}
            run={run}
          />
        ))}
      </div>
      {lastRun && (
        <div className="mt-3 rounded border border-border bg-bg px-3 py-2 text-2xs text-text-muted">
          <span className="font-mono text-text">{lastRun.job}</span>{" "}
          {lastRun.started ? "started" : "did not start"}
          {lastRun.report ? (
            <>
              {" "}
              — consolidated {lastRun.report.consolidated}, merged {lastRun.report.entities_merged}, archived{" "}
              {lastRun.report.archived}, pruned {lastRun.report.edges_pruned}
            </>
          ) : null}
        </div>
      )}
    </Panel>
  );
}

function TriggerCard({
  job,
  label,
  description,
  run,
}: {
  job: MaintenanceJobName;
  label: string;
  description: string;
  run: ReturnType<typeof useRunMaintenance>;
}) {
  const pending = run.isPending && run.variables === job;
  return (
    <div className="flex flex-col gap-2 rounded border border-border bg-bg-inset p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-text">{label}</span>
        <code className="font-mono text-2xs text-text-faint">{job}</code>
      </div>
      <p className="min-h-[28px] text-2xs leading-relaxed text-text-muted">{description}</p>
      <Button variant="primary" loading={pending} onClick={() => run.mutate(job)} className="self-start">
        trigger
      </Button>
    </div>
  );
}
