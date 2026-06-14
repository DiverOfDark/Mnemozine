/**
 * ScopeTreeNav — the hierarchical SCOPE TREE navigator (PRD §4.2, core redesign).
 *
 * Scope is now an ordered PATH (global -> project -> sub-scope...), so the flat
 * scope dropdown is replaced by a drill-in tree. Each node shows its segment label
 * and its rolled-up count (this scope + all descendants — the ancestor-composed
 * view a query at that scope sees). Selecting a node drives the GLOBAL scope filter
 * (useScope): the table then shows that scope's ancestor-composed memories.
 *
 * The tree comes from /memories/facets/scope-tree; counts and the no-leak roll-up
 * are computed on the backend. Page-local to the Memories screen.
 */

import { useState } from "react";
import type { ScopeTreeNode } from "@/api/types";
import { Loading, ErrorState } from "@/components/primitives";
import { cn } from "@/lib/cn";

interface ScopeTreeNavProps {
  root: ScopeTreeNode | undefined;
  isLoading: boolean;
  error: Error | null;
  /** The currently selected scope string (from useScope), or null = all scopes. */
  selected: string | null;
  /** Select a scope path; null clears the scope filter (all scopes). */
  onSelect: (path: string | null) => void;
  onRetry?: () => void;
}

function NodeRow({
  node,
  selected,
  onSelect,
}: {
  node: ScopeTreeNode;
  selected: string | null;
  onSelect: (path: string) => void;
}) {
  // Expand the path that contains the current selection by default.
  const onSelectedPath =
    selected != null && (selected === node.path || selected.startsWith(`${node.path}/`));
  const [open, setOpen] = useState(node.depth === 0 || onSelectedPath);
  const hasChildren = node.children.length > 0;
  const isSelected = selected === node.path;

  return (
    <li>
      <div
        className={cn(
          "group flex items-center gap-1 rounded px-1 py-0.5",
          isSelected ? "bg-bg-active" : "hover:bg-bg-hover",
        )}
        style={{ paddingLeft: `${node.depth * 12 + 4}px` }}
      >
        {hasChildren ? (
          <button
            type="button"
            aria-label={open ? "collapse" : "expand"}
            onClick={() => setOpen((v) => !v)}
            className="w-3 shrink-0 text-text-faint hover:text-text-muted"
          >
            {open ? "▾" : "▸"}
          </button>
        ) : (
          <span className="w-3 shrink-0 text-text-faint">·</span>
        )}
        <button
          type="button"
          onClick={() => onSelect(node.path)}
          title={`scope: ${node.path}`}
          className={cn(
            "flex min-w-0 flex-1 items-center justify-between gap-2 truncate text-left font-mono text-2xs",
            isSelected ? "text-accent" : "text-text-muted group-hover:text-text",
          )}
        >
          <span className="truncate">{node.segment}</span>
          <span className="shrink-0 tabular-nums text-text-faint">{node.total_count}</span>
        </button>
      </div>
      {hasChildren && open && (
        <ul>
          {node.children.map((child) => (
            <NodeRow
              key={child.path}
              node={child}
              selected={selected}
              onSelect={onSelect}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

export function ScopeTreeNav({
  root,
  isLoading,
  error,
  selected,
  onSelect,
  onRetry,
}: ScopeTreeNavProps) {
  return (
    <div className="flex h-full min-h-0 w-52 shrink-0 flex-col overflow-hidden rounded-md border border-border bg-bg-raised">
      <div className="flex items-center justify-between border-b border-border px-2 py-1.5">
        <span className="font-mono text-2xs uppercase tracking-wide text-text-muted">
          scope tree
        </span>
        {selected != null && (
          <button
            type="button"
            onClick={() => onSelect(null)}
            className="font-mono text-2xs text-accent hover:text-accent-hover"
            title="Clear scope filter (show all scopes)"
          >
            clear
          </button>
        )}
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-1">
        {isLoading ? (
          <Loading label="loading scopes…" />
        ) : error ? (
          <ErrorState error={error} onRetry={onRetry} />
        ) : root ? (
          <ul>
            <NodeRow node={root} selected={selected} onSelect={onSelect} />
          </ul>
        ) : (
          <div className="px-2 py-3 text-2xs text-text-faint">No scopes yet.</div>
        )}
      </div>
    </div>
  );
}
