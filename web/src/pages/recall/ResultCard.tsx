/**
 * ResultCard (FE-B / PRD §4.5) — one ranked recall hit: rank index, the relevance
 * <ScoreBar>, category/tier/status badges (color-by-category, struck/greyed if
 * superseded), the scope path, the content snippet, the **why-it-surfaced** note,
 * and a link into the memory detail. Keyboard-focusable; the parent list drives j/k
 * selection by setting `selected` + scrolling it into view.
 */

import { forwardRef } from "react";
import { Link } from "react-router-dom";
import {
  CategoryBadge,
  CrossRefBadge,
  ScopePath,
  ScoreBar,
  StatusBadge,
  TierBadge,
} from "@/components/index";
import type { ScoredMemory } from "@/api";
import { PATHS } from "@/routes";
import { cn } from "@/lib/cn";
import { shortId } from "@/lib/format";
import { categoryColor } from "@/theme/tokens";

interface ResultCardProps {
  rank: number;
  scored: ScoredMemory;
  selected: boolean;
  /** Normalized 0..1 value for the ScoreBar (caller normalizes raw scores > 1). */
  scoreNorm: number;
}

export const ResultCard = forwardRef<HTMLDivElement, ResultCardProps>(function ResultCard(
  { rank, scored, selected, scoreNorm },
  ref,
) {
  const { memory, score, why } = scored;
  const active = memory.active;
  return (
    <div
      ref={ref}
      data-rank={rank}
      className={cn(
        "rounded-md border bg-bg-raised transition-colors",
        selected ? "border-accent bg-bg-hover" : "border-border hover:border-border-strong",
      )}
      style={{ borderLeftWidth: 3, borderLeftColor: categoryColor(memory.category).fg }}
    >
      <div className="flex items-start gap-3 p-3">
        <span className="mt-0.5 w-6 shrink-0 text-right font-mono text-2xs tabular-nums text-text-faint">
          {rank}
        </span>
        <div className="min-w-0 flex-1">
          {/* badges + score row */}
          <div className="mb-1.5 flex flex-wrap items-center gap-1.5">
            <CategoryBadge category={memory.category} />
            {memory.cross_ref_candidate && <CrossRefBadge />}
            <ScopePath scope={memory.scope} />
            <TierBadge tier={memory.tier} />
            <StatusBadge active={active} />
            <span className="ml-auto flex items-center gap-2">
              <ScoreBar value={scoreNorm} format="decimal" width={80} showValue={false} />
              <span className="font-mono text-2xs tabular-nums text-text-muted" title="raw recall score">
                {score.toFixed(3)}
              </span>
            </span>
          </div>

          {/* content snippet */}
          <Link
            to={PATHS.memoryDetail(memory.id)}
            className={cn(
              "block font-mono text-xs leading-relaxed hover:underline",
              active ? "text-text" : "superseded",
            )}
            title="open memory detail"
          >
            {memory.content}
          </Link>

          {/* why-it-surfaced */}
          {why && (
            <p className="mt-1.5 border-l-2 border-border-strong pl-2 text-2xs italic text-text-muted">
              why: {why}
            </p>
          )}

          {/* meta */}
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 font-mono text-2xs text-text-faint">
            <span title={memory.id}>{shortId(memory.id)}</span>
            <span>scope: {memory.scope}</span>
            {memory.entities.length > 0 && <span>entities: {memory.entities.join(", ")}</span>}
          </div>
        </div>
      </div>
    </div>
  );
});
