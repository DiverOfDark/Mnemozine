/**
 * Sparkline — a tiny inline SVG area/line chart for the store-growth trend
 * (PRD §4.1). Page-local to the Dashboard. Pure presentation: the caller derives
 * the numeric series (e.g. cumulative writes per day from the activity feed) and
 * passes it in. Uses theme HEX values only where SVG can't take Tailwind classes.
 */

import { useId } from "react";
import { HEX } from "@/theme/tokens";

interface SparklineProps {
  /** The series to plot, oldest → newest. */
  values: number[];
  width?: number;
  height?: number;
  /** Line/area color (hex). Defaults to the accent token. */
  color?: string;
  className?: string;
}

export function Sparkline({
  values,
  width = 220,
  height = 48,
  color = HEX.tier.hot,
  className,
}: SparklineProps) {
  const gradId = useId();
  const pad = 2;
  const n = values.length;

  if (n < 2) {
    return (
      <div
        className="flex items-center justify-center text-2xs text-text-faint"
        style={{ width, height }}
      >
        not enough data
      </div>
    );
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const innerW = width - pad * 2;
  const innerH = height - pad * 2;

  const x = (i: number) => pad + (i / (n - 1)) * innerW;
  const y = (v: number) => pad + innerH - ((v - min) / span) * innerH;

  const linePoints = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const areaPath =
    `M ${x(0).toFixed(1)},${(height - pad).toFixed(1)} ` +
    values.map((v, i) => `L ${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ") +
    ` L ${x(n - 1).toFixed(1)},${(height - pad).toFixed(1)} Z`;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={className}
      role="img"
      aria-label="store growth sparkline"
      preserveAspectRatio="none"
    >
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.28} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={areaPath} fill={`url(#${gradId})`} />
      <polyline points={linePoints} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={x(n - 1)} cy={y(values[n - 1]!)} r={2.2} fill={color} />
    </svg>
  );
}
