/**
 * Column definitions for the Memories DataTable (PRD §4.2): category · content ·
 * scope · entities · confidence · tier · validity (valid_from → valid_to) ·
 * last_accessed · access_count, plus an active/superseded state badge. Page-local
 * to the Memories screen. Cells use the shared design-system components
 * (CategoryBadge / ScopePath / TierBadge / StatusBadge / ScoreBar) and the shared
 * formatters — never hand-rolled color spans.
 *
 * The category column shows the FREE-FORM category chip (+ a cross-ref seed flag);
 * the scope column renders the HIERARCHICAL scope path as a breadcrumb.
 *
 * The per-cell renderers are plain functions (not standalone components) so this
 * module stays a single constant export — keeping react-refresh / fast-reload happy.
 */

import type { ReactNode } from "react";
import type { Column } from "@/components/DataTable";
import {
  CategoryBadge,
  CrossRefBadge,
  ScopePath,
  TierBadge,
  StatusBadge,
  Badge,
} from "@/components/Badge";
import { ScoreBar } from "@/components/ScoreBar";
import type { MemoryListItem } from "@/api/types";
import { formatDate, formatRelative, formatDateTime } from "@/lib/format";

/** Category cell: the free-form category chip plus an optional cross-ref seed flag. */
function renderCategory(row: MemoryListItem): ReactNode {
  return (
    <div className="flex flex-wrap items-center gap-1">
      <CategoryBadge category={row.category} />
      {row.cross_ref_candidate && <CrossRefBadge />}
    </div>
  );
}

/** Entity chips, capped with a "+N" overflow marker. */
function renderEntities(entities: string[]): ReactNode {
  if (entities.length === 0) return <span className="text-text-faint">—</span>;
  const shown = entities.slice(0, 2);
  const rest = entities.length - shown.length;
  return (
    <div className="flex flex-wrap items-center gap-1">
      {shown.map((e) => (
        <Badge key={e} className="normal-case" title={e}>
          {e}
        </Badge>
      ))}
      {rest > 0 && (
        <span className="font-mono text-2xs text-text-faint" title={entities.join(", ")}>
          +{rest}
        </span>
      )}
    </div>
  );
}

/** Validity window cell — from date, and either "now" (active) or the closed date. */
function renderValidity(row: MemoryListItem): ReactNode {
  return (
    <div className="flex flex-col leading-tight">
      <span className="font-mono text-2xs text-text-muted" title={formatDateTime(row.valid_from)}>
        {formatDate(row.valid_from)}
      </span>
      {row.valid_to ? (
        <span className="font-mono text-2xs text-superseded" title={formatDateTime(row.valid_to)}>
          → {formatDate(row.valid_to)}
        </span>
      ) : (
        <span className="font-mono text-2xs text-active">→ now</span>
      )}
    </div>
  );
}

export const memoryColumns: Column<MemoryListItem>[] = [
  {
    id: "category",
    header: "Category",
    width: 140,
    cell: (row) => renderCategory(row),
  },
  {
    id: "content",
    header: "Content",
    className: "min-w-[240px] max-w-[440px]",
    cell: (row) => (
      <span className="block truncate text-xs text-text" title={row.content}>
        {row.content}
      </span>
    ),
  },
  {
    id: "scope",
    header: "Scope",
    width: 150,
    cell: (row) => <ScopePath scope={row.scope} />,
  },
  {
    id: "entities",
    header: "Entities",
    width: 160,
    cell: (row) => renderEntities(row.entities),
  },
  {
    id: "confidence",
    header: "Conf",
    width: 96,
    align: "right",
    cell: (row) => <ScoreBar value={row.confidence} format="decimal" width={44} />,
  },
  {
    id: "tier",
    header: "Tier",
    width: 86,
    cell: (row) => <TierBadge tier={row.tier} />,
  },
  {
    id: "state",
    header: "State",
    width: 104,
    cell: (row) => <StatusBadge active={row.active} />,
  },
  {
    id: "validity",
    header: "Validity",
    width: 110,
    cell: (row) => renderValidity(row),
  },
  {
    id: "last_accessed",
    header: "Accessed",
    width: 84,
    align: "right",
    cell: (row) => (
      <span className="font-mono text-2xs text-text-muted" title={formatDateTime(row.last_accessed)}>
        {formatRelative(row.last_accessed)}
      </span>
    ),
  },
  {
    id: "access_count",
    header: "Hits",
    width: 56,
    align: "right",
    cell: (row) => (
      <span className="font-mono text-xs tabular-nums text-text-muted">{row.access_count.toLocaleString()}</span>
    ),
  },
];
