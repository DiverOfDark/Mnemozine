/**
 * Store-growth panel (PRD §4.1). Page-local to the Dashboard. Wraps the local
 * Sparkline with a derived growth series (useGrowthSeries) and a small legend.
 * The curve is backed by GET /api/stats/growth — a real, retroactive grouped count
 * of memories by creation day — so it renders the genuine trend with no caveats; it
 * only falls back to the Sparkline's "not enough data" state for an empty store.
 */

import { Panel } from "@/components/primitives";
import { Sparkline } from "@/pages/dashboard/Sparkline";
import { useGrowthSeries } from "@/pages/dashboard/useGrowthSeries";
import { HEX } from "@/theme/tokens";

export function GrowthPanel({ scope, windowDays = 14 }: { scope: string | null; windowDays?: number }) {
  const series = useGrowthSeries(scope, windowDays);
  const last = series.cumulative.at(-1) ?? 0;

  return (
    <Panel
      title="Store growth"
      actions={<span className="font-mono text-2xs text-text-faint">last {windowDays}d</span>}
    >
      <div className="flex items-center justify-between gap-4">
        <div className="flex flex-col gap-0.5">
          <span className="font-mono text-2xl tabular-nums leading-none text-text">
            +{series.total.toLocaleString()}
          </span>
          <span className="text-2xs uppercase tracking-wide text-text-faint">writes / {windowDays}d</span>
        </div>
        <div className="shrink-0">
          {series.empty ? (
            <Sparkline values={[]} color={HEX.tier.hot} />
          ) : (
            <Sparkline values={series.cumulative} color={HEX.tier.hot} />
          )}
          <div className="mt-1 flex justify-between font-mono text-2xs text-text-faint">
            <span>{series.days.at(0)?.slice(5) ?? ""}</span>
            <span className="tabular-nums">Σ {last.toLocaleString()}</span>
            <span>{series.days.at(-1)?.slice(5) ?? ""}</span>
          </div>
        </div>
      </div>
    </Panel>
  );
}
