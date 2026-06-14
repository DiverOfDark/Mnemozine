/**
 * Logs screen — local filter state hook (FE-C / PRD §4.6).
 *
 * Owns the ActivityQuery the Logs feed sends to useActivity, plus the paging
 * offset. Kept page-local (not in shared state) because no other screen needs the
 * activity filter shape. The top-bar scope is folded in by the page (it maps to the
 * `project` filter for project scopes).
 */

import { useCallback, useMemo, useState } from "react";
import type { ActivityKind, ActivityQuery } from "@/api";
import { parseScope } from "@/lib/format";

/** The mutable filter fields the toolbar edits (paging is handled separately). */
export interface ActivityFilterState {
  kinds: ActivityKind[];
  source: string;
  sessionId: string;
  refMemoryId: string;
  since: string;
  until: string;
}

const EMPTY_FILTERS: ActivityFilterState = {
  kinds: [],
  source: "",
  sessionId: "",
  refMemoryId: "",
  since: "",
  until: "",
};

export const ACTIVITY_PAGE_SIZE = 100;

export interface UseActivityFiltersResult {
  filters: ActivityFilterState;
  /** The composed ActivityQuery (filters + scope-derived project + paging). */
  query: ActivityQuery;
  offset: number;
  setKinds: (kinds: ActivityKind[]) => void;
  toggleKind: (kind: ActivityKind) => void;
  setField: (field: keyof ActivityFilterState, value: string) => void;
  setOffset: (offset: number) => void;
  reset: () => void;
  /** True when any filter (other than scope/paging) is active. */
  hasActiveFilters: boolean;
}

/** Convert an ISO-local `datetime-local` value to an ISO-8601 string, if present. */
function toIso(value: string): string | undefined {
  if (!value) return undefined;
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return undefined;
  return d.toISOString();
}

export function useActivityFilters(scope: string | null): UseActivityFiltersResult {
  const [filters, setFilters] = useState<ActivityFilterState>(EMPTY_FILTERS);
  const [offset, setOffset] = useState(0);

  const setKinds = useCallback((kinds: ActivityKind[]) => {
    setFilters((f) => ({ ...f, kinds }));
    setOffset(0);
  }, []);

  const toggleKind = useCallback((kind: ActivityKind) => {
    setFilters((f) => {
      const next = f.kinds.includes(kind)
        ? f.kinds.filter((k) => k !== kind)
        : [...f.kinds, kind];
      return { ...f, kinds: next };
    });
    setOffset(0);
  }, []);

  const setField = useCallback((field: keyof ActivityFilterState, value: string) => {
    setFilters((f) => ({ ...f, [field]: value }));
    setOffset(0);
  }, []);

  const reset = useCallback(() => {
    setFilters(EMPTY_FILTERS);
    setOffset(0);
  }, []);

  const query = useMemo<ActivityQuery>(() => {
    const parsed = parseScope(scope);
    return {
      kind: filters.kinds.length ? filters.kinds : undefined,
      source: filters.source.trim() || undefined,
      session_id: filters.sessionId.trim() || undefined,
      ref_memory_id: filters.refMemoryId.trim() || undefined,
      project: parsed.kind === "project" ? parsed.project : undefined,
      since: toIso(filters.since),
      until: toIso(filters.until),
      limit: ACTIVITY_PAGE_SIZE,
      offset,
    };
  }, [filters, scope, offset]);

  const hasActiveFilters = useMemo(
    () =>
      filters.kinds.length > 0 ||
      Boolean(filters.source.trim()) ||
      Boolean(filters.sessionId.trim()) ||
      Boolean(filters.refMemoryId.trim()) ||
      Boolean(filters.since) ||
      Boolean(filters.until),
    [filters],
  );

  return {
    filters,
    query,
    offset,
    setKinds,
    toggleKind,
    setField,
    setOffset,
    reset,
    hasActiveFilters,
  };
}
