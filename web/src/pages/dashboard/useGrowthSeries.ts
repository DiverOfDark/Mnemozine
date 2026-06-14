/**
 * useGrowthSeries — derives the Dashboard's store-growth sparkline (PRD §4.1) from
 * the activity log. There is no dedicated growth endpoint; the honest approximation
 * is to bucket recent write-producing activity events (ingest / extract_decision)
 * by day and accumulate. When the activity log is disabled the series is empty and
 * the Sparkline renders its "not enough data" state — never a fabricated trend.
 *
 * Page-local to the Dashboard.
 */

import { useMemo } from "react";
import { useActivity } from "@/api/hooks";
import type { ActivityEventOut } from "@/api/types";

export interface GrowthSeries {
  /** Cumulative write count per day bucket, oldest → newest. */
  cumulative: number[];
  /** Per-day write counts, oldest → newest. */
  daily: number[];
  /** Day labels (YYYY-MM-DD), aligned with the series. */
  days: string[];
  /** Total writes observed in the window. */
  total: number;
  isLoading: boolean;
  error: unknown;
  /** True when the underlying activity log returned nothing (likely disabled). */
  empty: boolean;
}

const DAY_MS = 24 * 60 * 60 * 1000;
/** Events that represent a write into the store (drive the growth curve). */
const WRITE_KINDS = new Set<ActivityEventOut["kind"]>(["ingest", "extract_decision"]);

function dayKey(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export function useGrowthSeries(scope: string | null, windowDays = 14): GrowthSeries {
  const project = scope && scope !== "global" ? scope.replace(/^project:/, "") : undefined;
  const since = useMemo(() => new Date(Date.now() - windowDays * DAY_MS).toISOString(), [windowDays]);

  const { data, isLoading, error } = useActivity(
    {
      kind: ["ingest", "extract_decision"],
      since,
      limit: 500,
      ...(project ? { project } : {}),
    },
    { refetchInterval: 60_000 },
  );

  return useMemo<GrowthSeries>(() => {
    const events = data?.items ?? [];
    const writes = events.filter((e) => WRITE_KINDS.has(e.kind));

    // Build an empty bucket per day across the window so the curve has a stable x-axis.
    const days: string[] = [];
    const buckets = new Map<string, number>();
    const start = new Date(Date.now() - (windowDays - 1) * DAY_MS);
    for (let i = 0; i < windowDays; i += 1) {
      const d = new Date(start.getTime() + i * DAY_MS);
      const key = dayKey(d);
      days.push(key);
      buckets.set(key, 0);
    }

    for (const e of writes) {
      const key = dayKey(new Date(e.ts));
      if (buckets.has(key)) buckets.set(key, (buckets.get(key) ?? 0) + 1);
    }

    const daily = days.map((k) => buckets.get(k) ?? 0);
    let running = 0;
    const cumulative = daily.map((v) => (running += v));

    return {
      cumulative,
      daily,
      days,
      total: writes.length,
      isLoading,
      error,
      empty: !isLoading && !error && writes.length === 0,
    };
  }, [data, isLoading, error, windowDays]);
}
