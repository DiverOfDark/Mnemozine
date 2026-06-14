/**
 * Maintenance / Ops (PRD §4.7) — the operator's control room for the memory layer.
 *
 * Three sections, tab-switched so each gets the full height of the dense console:
 *   1. Scheduler   — scheduler running state + cron, per-job last/next runs and
 *                    last-report counts, and trigger-job buttons (consolidate, decay,
 *                    entity-resolution, audit, migrate-index).  [useMaintenance,
 *                    useRunMaintenance]
 *   2. Merge review — entity-resolution HITL merge-candidate queue.  [useMergeCandidates,
 *                    useRunMaintenance]
 *   3. Suppression — cross-reference suppression-list management.  [useCrossRefs,
 *                    useSuppressCrossRef]  (scoped from the top-bar ScopeContext)
 *
 * This file owns only its page + the page-local components under pages/maintenance/**.
 * It consumes shared design-system components and the typed api hooks; it edits no
 * shared/contract files.
 */

import { useState } from "react";

import { useMaintenance } from "@/api";
import { Page, Badge } from "@/components";
import { SchedulerPanel } from "@/pages/maintenance/SchedulerPanel";
import { MergeCandidates } from "@/pages/maintenance/MergeCandidates";
import { SuppressionList } from "@/pages/maintenance/SuppressionList";

type Tab = "scheduler" | "merge" | "suppression";

const TABS: { id: Tab; label: string }[] = [
  { id: "scheduler", label: "Scheduler & jobs" },
  { id: "merge", label: "Merge review" },
  { id: "suppression", label: "Suppression list" },
];

export default function Maintenance() {
  const [tab, setTab] = useState<Tab>("scheduler");
  const maintenance = useMaintenance({ refetchInterval: 15_000 });

  const runningChip = maintenance.data ? (
    <Badge
      textClass={maintenance.data.scheduler_running ? "text-ok" : "text-text-faint"}
      bgClass={maintenance.data.scheduler_running ? "bg-tier-bg-hot" : "bg-bg-inset"}
      dotClass={maintenance.data.scheduler_running ? "bg-ok" : "bg-text-faint"}
    >
      scheduler {maintenance.data.scheduler_running ? "running" : "stopped"}
    </Badge>
  ) : null;

  return (
    <Page
      title="Maintenance / Ops"
      subtitle="Scheduler · trigger jobs · entity-resolution review · suppression list"
      actions={
        <div className="flex items-center gap-3">
          {runningChip}
          <TabBar tab={tab} onChange={setTab} />
        </div>
      }
      bodyClassName="flex flex-col"
    >
      {tab === "scheduler" && (
        <SchedulerPanel
          status={maintenance.data}
          isLoading={maintenance.isLoading}
          error={maintenance.error}
          onRetry={() => void maintenance.refetch()}
        />
      )}
      {tab === "merge" && <MergeCandidates />}
      {tab === "suppression" && <SuppressionList />}
    </Page>
  );
}

function TabBar({ tab, onChange }: { tab: Tab; onChange: (t: Tab) => void }) {
  return (
    <div className="flex items-center gap-0.5 rounded border border-border bg-bg-inset p-0.5">
      {TABS.map((t) => (
        <button
          key={t.id}
          type="button"
          onClick={() => onChange(t.id)}
          className={
            "rounded px-2.5 py-1 text-xs font-medium transition-colors duration-fast " +
            (tab === t.id ? "bg-bg-active text-text" : "text-text-muted hover:bg-bg-hover hover:text-text")
          }
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
