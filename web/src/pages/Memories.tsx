/**
 * Memories (PRD §4.2, the core screen) — the filterable / sortable, keyboard-driven
 * table over every memory. Columns: category · content · scope · entities ·
 * confidence · tier · validity · last_accessed · access_count + active/superseded
 * state. Filters: a DYNAMIC category facet, tier, active-vs-superseded, entity,
 * source, free-text (q) — all URL-backed — layered over a SCOPE TREE navigator that
 * drives the top-bar scope (useScope). Superseded rows render struck + greyed via
 * the shared `.superseded` treatment. Activating a row (click / Enter) opens the
 * Memory detail route.
 *
 * The core data-model redesign surfaces here: the fixed `type` filter is replaced
 * by the discovered-category facet (useCategoryFacets) and the flat scope picker by
 * the hierarchical scope tree (useScopeTree); selecting a scope node shows that
 * scope's ancestor-composed memories.
 *
 * Composes the shared DataTable + badges + score bar with the typed useMemories hook
 * via page-local pieces under src/pages/memories/. Owns ONLY this file + that folder;
 * never touches the router, the api client/hooks, the theme, or shared components.
 */

import { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { PageHeader } from "@/components/AppShell";
import { DataTable } from "@/components/DataTable";
import { KeyboardHints } from "@/components/KeyboardHints";
import { useCategoryFacets, useMemories, useScopeTree } from "@/api/hooks";
import { useScope } from "@/state/scope";
import { PATHS } from "@/routes";
import type { MemoryListItem } from "@/api/types";
import { cn } from "@/lib/cn";

import { useMemoriesFilters, PAGE_SIZE } from "@/pages/memories/useMemoriesFilters";
import { FilterBar } from "@/pages/memories/FilterBar";
import { memoryColumns } from "@/pages/memories/columns";
import { Pagination } from "@/pages/memories/Pagination";
import { ScopeTreeNav } from "@/pages/memories/ScopeTreeNav";

const KEY_HINTS = [
  { keys: ["j"], label: "down" },
  { keys: ["k"], label: "up" },
  { keys: ["↵"], label: "open detail" },
];

export default function Memories() {
  const navigate = useNavigate();
  const { scope, setScope } = useScope();
  const { filters, query, setFilters, clearAll, setOffset, hasActiveFilters, page } =
    useMemoriesFilters(scope);

  const { data, isLoading, error, refetch, isFetching } = useMemories(query, {
    placeholderData: (prev) => prev, // keep prior page visible while the next page loads
  });

  // Discovery surfaces for the open-ended model: the dynamic category facet and the
  // hierarchical scope tree (both read-only; they only drive the filters).
  const facets = useCategoryFacets();
  const scopeTree = useScopeTree();

  const rows = data?.items ?? [];
  const pageInfo = data?.page;

  const openRow = useCallback(
    (row: MemoryListItem) => navigate(PATHS.memoryDetail(row.id)),
    [navigate],
  );

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        title="Memories"
        subtitle="filterable memory table · scope tree → drill · row → detail"
        actions={<KeyboardHints hints={KEY_HINTS} />}
      />

      <div className="flex min-h-0 flex-1 gap-3 p-5">
        <ScopeTreeNav
          root={scopeTree.data?.root}
          isLoading={scopeTree.isLoading}
          error={scopeTree.error}
          selected={scope}
          onSelect={(path) => setScope(path === "global" ? null : path)}
          onRetry={() => void scopeTree.refetch()}
        />

        <div className="flex min-h-0 flex-1 flex-col gap-3">
          <FilterBar
            filters={filters}
            scope={scope}
            setFilters={setFilters}
            clearAll={clearAll}
            hasActiveFilters={hasActiveFilters}
            total={pageInfo?.total}
            categoryFacets={facets.data?.facets}
          />

          <div
            className={cn(
              "flex min-h-0 flex-1 flex-col overflow-hidden rounded-md border border-border bg-bg-raised transition-opacity",
              isFetching && !isLoading && "opacity-70",
            )}
          >
            <DataTable<MemoryListItem>
              className="min-h-0 flex-1"
              columns={memoryColumns}
              rows={rows}
              rowKey={(row) => row.id}
              onRowActivate={openRow}
              rowClassName={(row) => (row.active ? undefined : "superseded")}
              isLoading={isLoading}
              error={error}
              onRetry={() => void refetch()}
              emptyTitle={hasActiveFilters ? "No memories match these filters" : "No memories yet"}
              emptyHint={
                hasActiveFilters
                  ? "Loosen or clear the filters above to widen the result set."
                  : "Memories appear here as the pipeline ingests and extracts them."
              }
            />
            <Pagination
              page={pageInfo}
              count={rows.length}
              onPrev={() => setOffset(Math.max(0, page.offset - PAGE_SIZE))}
              onNext={() => setOffset(page.offset + PAGE_SIZE)}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
