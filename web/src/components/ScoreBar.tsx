/**
 * ScoreBar — a compact horizontal relevance/confidence bar (PRD §4.5 recall scores,
 * §4.2 confidence column). Color follows the low→mid→high score gradient token.
 */

import { scoreColor } from "@/theme/tokens";
import { cn } from "@/lib/cn";

interface ScoreBarProps {
  /** Value in [0, 1] (clamped). Recall scores above 1 should be normalized by the caller. */
  value: number;
  /** Show the numeric value alongside the bar. */
  showValue?: boolean;
  /** Format for the trailing number: "percent" (87%) or "decimal" (0.87). */
  format?: "percent" | "decimal";
  width?: number | string;
  className?: string;
}

export function ScoreBar({
  value,
  showValue = true,
  format = "decimal",
  width = 64,
  className,
}: ScoreBarProps) {
  const v = Math.max(0, Math.min(1, value));
  const color = scoreColor(v);
  const label = format === "percent" ? `${Math.round(v * 100)}%` : v.toFixed(2);
  return (
    <div className={cn("inline-flex items-center gap-2", className)} title={`score ${label}`}>
      <div
        className="h-1.5 overflow-hidden rounded-full bg-bg-inset"
        style={{ width: typeof width === "number" ? `${width}px` : width }}
      >
        <div
          className="h-full rounded-full transition-all duration-fast"
          style={{ width: `${v * 100}%`, backgroundColor: color }}
        />
      </div>
      {showValue && <span className="font-mono text-2xs text-text-muted tabular-nums">{label}</span>}
    </div>
  );
}
