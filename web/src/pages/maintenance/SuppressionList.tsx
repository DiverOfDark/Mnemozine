/**
 * SuppressionList — cross-reference suppression management (PRD §4.7, R2).
 *
 * Lists surfaced cross-references for the active scope (via useCrossRefs, scoped from
 * the top-bar ScopeContext). Each row shows the cross-referenced memory, its
 * human-readable `reason` (FR-RET-6), the shared entities that triggered the link,
 * and its suppression state. The operator can suppress an active cross-reference
 * (useSuppressCrossRef with its context_key) to stop it surfacing; suppressed rows
 * are rendered greyed/struck via the shared `.superseded` treatment.
 *
 * "Show suppressed" toggles include_suppressed so the dismissed list itself is
 * reviewable. Page-local: design-system components + typed api hooks only.
 */

import { useState } from "react";
import { Link } from "react-router-dom";

import {
  useCrossRefs,
  useSuppressCrossRef,
  type CrossRefItem,
} from "@/api";
import { Badge, Button, CategoryBadge, DataTable, ScoreBar, type Column } from "@/components";
import { useScope } from "@/state/scope";
import { parseScope, shortId } from "@/lib/format";

export function SuppressionList() {
  const { scope } = useScope();
  const [showSuppressed, setShowSuppressed] = useState(true);

  const parsed = parseScope(scope);
  const project = parsed.kind === "project" ? parsed.project : undefined;

  const { data, isLoading, error, refetch, isRefetching } = useCrossRefs({
    project,
    include_suppressed: showSuppressed,
    limit: 100,
  });
  const suppress = useSuppressCrossRef();

  const items = data?.items ?? [];
  const suppressedCount = items.filter((i) => i.suppressed).length;

  const columns: Column<CrossRefItem>[] = [
    {
      id: "memory",
      header: "Cross-referenced memory",
      cell: (row) => (
        <Link
          to={`/memories/${row.memory.id}`}
          className="flex flex-col gap-0.5 hover:text-accent"
          onClick={(e) => e.stopPropagation()}
        >
          <span className={row.suppressed ? "superseded line-clamp-2" : "line-clamp-2 text-text"}>
            {row.memory.content}
          </span>
          <span className="font-mono text-2xs text-text-faint">{shortId(row.memory.id)}</span>
        </Link>
      ),
    },
    {
      id: "category",
      header: "Category",
      width: 140,
      cell: (row) => <CategoryBadge category={row.memory.category} />,
    },
    {
      id: "reason",
      header: "Reason",
      cell: (row) => (
        <span className="text-2xs leading-relaxed text-text-muted" title={row.reason}>
          {row.reason}
        </span>
      ),
    },
    {
      id: "shared",
      header: "Shared entities",
      width: 180,
      cell: (row) =>
        row.shared_entities.length === 0 ? (
          <span className="text-text-faint">—</span>
        ) : (
          <div className="flex flex-wrap gap-1">
            {row.shared_entities.slice(0, 4).map((ent) => (
              <Badge key={ent} textClass="text-crossref" bgClass="bg-bg-inset">
                {ent}
              </Badge>
            ))}
            {row.shared_entities.length > 4 && (
              <span className="text-2xs text-text-faint">+{row.shared_entities.length - 4}</span>
            )}
          </div>
        ),
    },
    {
      id: "score",
      header: "Score",
      width: 96,
      cell: (row) => <ScoreBar value={row.score} width={48} />,
    },
    {
      id: "action",
      header: "",
      width: 110,
      align: "right",
      cell: (row) => {
        if (row.suppressed) {
          return (
            <Badge textClass="text-superseded" bgClass="bg-tier-bg-archive" dotClass="bg-superseded">
              suppressed
            </Badge>
          );
        }
        const pending = suppress.isPending && suppress.variables?.memoryId === row.memory.id;
        const contextKey = row.context_key;
        return (
          <Button
            variant="danger"
            disabled={!contextKey}
            loading={pending}
            title={contextKey ? "dismiss this cross-reference" : "no context key — cannot suppress"}
            onClick={(e) => {
              e.stopPropagation();
              if (contextKey) {
                suppress.mutate({ memoryId: row.memory.id, body: { context_key: contextKey } });
              }
            }}
          >
            suppress
          </Button>
        );
      },
    },
  ];

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-2xs leading-relaxed text-text-muted">
          Manage dismissed cross-references for{" "}
          <span className="font-mono text-text">{scope ?? "all scopes"}</span>. Suppressing a connection stops it
          from surfacing in recall / the graph overlay.
          {suppressedCount > 0 && (
            <>
              {" "}
              <span className="text-superseded">{suppressedCount} currently suppressed</span>.
            </>
          )}
        </p>
        <div className="flex shrink-0 items-center gap-2">
          <label className="flex cursor-pointer items-center gap-1.5 text-2xs text-text-muted">
            <input
              type="checkbox"
              checked={showSuppressed}
              onChange={(e) => setShowSuppressed(e.target.checked)}
              className="accent-accent"
            />
            show suppressed
          </label>
          <Button variant="ghost" loading={isRefetching} onClick={() => void refetch()}>
            refresh
          </Button>
        </div>
      </div>

      {suppress.isError && (
        <div className="rounded border border-danger/40 bg-danger/5 px-3 py-1.5 text-2xs text-danger">
          suppress failed: {suppress.error instanceof Error ? suppress.error.message : String(suppress.error)}
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-hidden rounded-md border border-border bg-bg-raised">
        <DataTable
          columns={columns}
          rows={items}
          rowKey={(r) => r.memory.id + (r.context_key ?? "")}
          isLoading={isLoading}
          error={error}
          onRetry={() => void refetch()}
          keyboardNav={false}
          rowClassName={(r) => (r.suppressed ? "opacity-60" : undefined)}
          emptyTitle="No cross-references"
          emptyHint="No surfaced cross-references for this scope. Cross-references appear when memories share entities across projects."
        />
      </div>
    </div>
  );
}
