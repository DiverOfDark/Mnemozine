/**
 * MergeCandidates — entity-resolution HITL review (PRD §4.7, FR-MNT-4).
 *
 * Lists the merge candidates surfaced by the resolver: source ↔ target entity pair,
 * a similarity ScoreBar, and the count of shared graph neighbors (the evidence the
 * two names co-refer). Higher similarity / more shared neighbors = greyer confidence
 * the merge is safe. Each candidate links its entities into the Graph explorer for a
 * deeper look.
 *
 * The actual merge is performed by the entity-resolution maintenance job (there is no
 * per-candidate merge endpoint in the contract), so the panel's primary action is to
 * trigger that job once the operator has reviewed the queue — see integration_notes.
 *
 * Page-local: design-system components + typed api hooks only.
 */

import { Link } from "react-router-dom";

import { useMergeCandidates, useRunMaintenance, type MergeCandidate } from "@/api";
import { Badge, Button, DataTable, ScoreBar, type Column } from "@/components";
import { shortId } from "@/lib/format";

export function MergeCandidates() {
  const { data, isLoading, error, refetch, isRefetching } = useMergeCandidates();
  const run = useRunMaintenance();
  const candidates = data?.candidates ?? [];

  const columns: Column<MergeCandidate>[] = [
    {
      id: "pair",
      header: "Candidate pair",
      cell: (row) => (
        <div className="flex items-center gap-2">
          <EntityChip id={row.source_id} name={row.source_name} />
          <span className="text-text-faint">↔</span>
          <EntityChip id={row.target_id} name={row.target_name} />
        </div>
      ),
    },
    {
      id: "similarity",
      header: "Similarity",
      width: 150,
      cell: (row) => <ScoreBar value={row.similarity} format="percent" width={72} />,
    },
    {
      id: "shared",
      header: "Shared neighbors",
      width: 140,
      align: "right",
      cell: (row) => (
        <Badge
          textClass={row.shared_neighbors > 0 ? "text-text" : "text-text-faint"}
          bgClass="bg-bg-inset"
          title="entities both candidates are connected to"
        >
          {row.shared_neighbors}
        </Badge>
      ),
    },
  ];

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="flex items-center justify-between gap-3">
        <p className="text-2xs leading-relaxed text-text-muted">
          Review co-referent entity pairs before merging. The merge is applied by the{" "}
          <span className="font-mono text-text">entity-resolution</span> job — trigger it once the queue looks
          correct.
        </p>
        <div className="flex shrink-0 items-center gap-2">
          <Button variant="ghost" loading={isRefetching} onClick={() => void refetch()}>
            refresh
          </Button>
          <Button
            variant="primary"
            loading={run.isPending && run.variables === "entity-resolution"}
            disabled={candidates.length === 0}
            onClick={() => run.mutate("entity-resolution")}
          >
            run entity-resolution
          </Button>
        </div>
      </div>

      {run.isError && (
        <div className="rounded border border-danger/40 bg-danger/5 px-3 py-1.5 text-2xs text-danger">
          run failed: {run.error instanceof Error ? run.error.message : String(run.error)}
        </div>
      )}
      {run.isSuccess && run.data.job === "entity-resolution" && (
        <div className="rounded border border-border bg-bg px-3 py-1.5 text-2xs text-text-muted">
          entity-resolution {run.data.started ? "ran" : "did not start"}
          {run.data.report ? <> — {run.data.report.entities_merged} merged</> : null}
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-hidden rounded-md border border-border bg-bg-raised">
        <DataTable
          columns={columns}
          rows={candidates}
          rowKey={(r) => `${r.source_id}:${r.target_id}`}
          isLoading={isLoading}
          error={error}
          onRetry={() => void refetch()}
          keyboardNav={false}
          emptyTitle="No merge candidates"
          emptyHint="The resolver found no co-referent entity pairs to review."
        />
      </div>
    </div>
  );
}

function EntityChip({ id, name }: { id: string; name: string }) {
  return (
    <Link
      to={`/graph?entity=${encodeURIComponent(name)}`}
      className="inline-flex items-center gap-1.5 rounded border border-border-strong bg-bg-inset px-1.5 py-0.5 text-xs text-text hover:border-accent hover:text-accent"
      title={`open ${name} in graph (${id})`}
    >
      <span className="truncate max-w-[180px]">{name}</span>
      <span className="font-mono text-2xs text-text-faint">{shortId(id, 6)}</span>
    </Link>
  );
}
