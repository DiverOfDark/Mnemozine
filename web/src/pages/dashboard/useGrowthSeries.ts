/**
 * useGrowthSeries — derives the Dashboard's store-growth sparkline (PRD §4.1) from
 * the dedicated GET /api/stats/growth endpoint. The server returns a dense,
 * zero-filled, oldest-first series of memories created per day (grouped by
 * valid_from) over the trailing window — real retroactive data, NOT a
 * reconstruction from the activity log. The series is "empty" only when the store
 * itself holds no memories in the window; in that case the Sparkline renders its
 * "not enough data" state rather than a fabricated trend.
 *
 * Page-local to the Dashboard.
 */

import { useMemo } from "react";
import { useGrowth } from "@/api/hooks";

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
  /** True when the store held no memories in the window (a genuinely empty store). */
  empty: boolean;
}

export function useGrowthSeries(scope: string | null, windowDays = 14): GrowthSeries {
  const { data, isLoading, error } = useGrowth(scope, windowDays, {
    refetchInterval: 60_000,
  });

  return useMemo<GrowthSeries>(() => {
    const days = data?.days ?? [];
    const daily = data?.daily ?? [];
    const cumulative = data?.cumulative ?? [];
    const total = data?.total ?? 0;

    return {
      cumulative,
      daily,
      days,
      total,
      isLoading,
      error,
      empty: !isLoading && !error && total === 0,
    };
  }, [data, isLoading, error]);
}
