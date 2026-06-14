/**
 * Pagination — the Memories table footer pager (PRD §4.2). Page-local. Drives the
 * limit/offset window of useMemoriesFilters off the `page` envelope returned by the
 * list response. Pure presentation: the page owns the offset state.
 */

import { Button } from "@/components/primitives";
import type { Page } from "@/api/types";

export function Pagination({
  page,
  count,
  onPrev,
  onNext,
}: {
  page: Page | undefined;
  /** Rows currently rendered (for the "showing X" label). */
  count: number;
  onPrev: () => void;
  onNext: () => void;
}) {
  if (!page) return null;
  const { total, limit, offset } = page;
  const from = total === 0 ? 0 : offset + 1;
  const to = offset + count;
  const hasPrev = offset > 0;
  const hasNext = offset + limit < total;
  const pageNum = Math.floor(offset / limit) + 1;
  const pageCount = Math.max(1, Math.ceil(total / limit));

  return (
    <div className="flex shrink-0 items-center justify-between border-t border-border px-1 py-2">
      <span className="font-mono text-2xs text-text-faint tabular-nums">
        {from.toLocaleString()}–{to.toLocaleString()} of {total.toLocaleString()}
      </span>
      <div className="flex items-center gap-2">
        <span className="font-mono text-2xs text-text-faint tabular-nums">
          page {pageNum} / {pageCount}
        </span>
        <Button variant="ghost" onClick={onPrev} disabled={!hasPrev}>
          ← prev
        </Button>
        <Button variant="ghost" onClick={onNext} disabled={!hasNext}>
          next →
        </Button>
      </div>
    </div>
  );
}
