/**
 * DataTable — the dense, dark, keyboard-navigable table used by the Memories,
 * Activity, Cross-refs and Eval screens. Generic over a row type; columns declare
 * how to render each cell. Superseded rows are greyed (rowClassName hook). Supports
 * j/k row navigation and Enter-to-open when `onRowActivate` is provided.
 *
 * Screen agents supply `columns` + `rows`; they should NOT restyle the table shell.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { cn } from "@/lib/cn";
import { Loading, EmptyState, ErrorState } from "@/components/primitives";

export interface Column<Row> {
  /** Stable column id. */
  id: string;
  /** Header label. */
  header: ReactNode;
  /** Cell renderer. */
  cell: (row: Row, index: number) => ReactNode;
  /** Optional fixed width (px) or tailwind width class via className. */
  width?: number | string;
  /** Tailwind classes applied to both header and cells (alignment etc). */
  className?: string;
  /** Right-align numeric columns. */
  align?: "left" | "right" | "center";
}

interface DataTableProps<Row> {
  columns: Column<Row>[];
  rows: Row[];
  /** Stable key extractor. */
  rowKey: (row: Row, index: number) => string;
  /** Click / Enter on a row. */
  onRowActivate?: (row: Row, index: number) => void;
  /** Highlight the currently-selected row (e.g. open in drawer). */
  selectedKey?: string | null;
  /** Extra classes per row (e.g. greyed superseded rows). */
  rowClassName?: (row: Row, index: number) => string | undefined;
  isLoading?: boolean;
  error?: unknown;
  onRetry?: () => void;
  emptyTitle?: string;
  emptyHint?: ReactNode;
  /** Enable j/k keyboard navigation (default true). */
  keyboardNav?: boolean;
  className?: string;
}

export function DataTable<Row>({
  columns,
  rows,
  rowKey,
  onRowActivate,
  selectedKey,
  rowClassName,
  isLoading,
  error,
  onRetry,
  emptyTitle = "No rows",
  emptyHint,
  keyboardNav = true,
  className,
}: DataTableProps<Row>) {
  const [cursor, setCursor] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const move = useCallback(
    (delta: number) => {
      setCursor((c) => Math.max(0, Math.min(rows.length - 1, c + delta)));
    },
    [rows.length],
  );

  useEffect(() => {
    if (!keyboardNav) return;
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      // Don't hijack typing in inputs.
      if (target.matches("input, textarea, select, [contenteditable='true']")) return;
      if (e.key === "j" || e.key === "ArrowDown") {
        e.preventDefault();
        move(1);
      } else if (e.key === "k" || e.key === "ArrowUp") {
        e.preventDefault();
        move(-1);
      } else if (e.key === "Enter" && onRowActivate && rows[cursor]) {
        e.preventDefault();
        onRowActivate(rows[cursor]!, cursor);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [keyboardNav, move, cursor, rows, onRowActivate]);

  if (isLoading) return <Loading />;
  if (error) return <ErrorState error={error} onRetry={onRetry} />;
  if (rows.length === 0) return <EmptyState title={emptyTitle} hint={emptyHint} />;

  return (
    <div ref={containerRef} className={cn("overflow-auto", className)}>
      <table className="w-full border-collapse text-xs">
        <thead className="sticky top-0 z-10 bg-bg-inset">
          <tr className="border-b border-border">
            {columns.map((col) => (
              <th
                key={col.id}
                style={col.width ? { width: typeof col.width === "number" ? `${col.width}px` : col.width } : undefined}
                className={cn(
                  "select-none px-3 py-2 text-left text-2xs font-medium uppercase tracking-wide text-text-faint",
                  col.align === "right" && "text-right",
                  col.align === "center" && "text-center",
                  col.className,
                )}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const key = rowKey(row, index);
            const isSelected = selectedKey != null && key === selectedKey;
            const isCursor = keyboardNav && index === cursor;
            return (
              <tr
                key={key}
                onClick={() => onRowActivate?.(row, index)}
                onMouseEnter={() => keyboardNav && setCursor(index)}
                className={cn(
                  "border-b border-border/60 transition-colors",
                  onRowActivate && "cursor-pointer",
                  isSelected ? "bg-bg-active" : isCursor ? "bg-bg-hover" : "hover:bg-bg-hover",
                  rowClassName?.(row, index),
                )}
              >
                {columns.map((col) => (
                  <td
                    key={col.id}
                    className={cn(
                      "px-3 py-2 align-middle text-text",
                      col.align === "right" && "text-right",
                      col.align === "center" && "text-center",
                      col.className,
                    )}
                  >
                    {col.cell(row, index)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
