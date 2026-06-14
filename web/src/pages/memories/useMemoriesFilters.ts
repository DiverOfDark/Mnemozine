/**
 * useMemoriesFilters — the Memories screen's URL-backed filter + pagination state
 * (PRD §4.2). All table filters (type / tier / entity / source / active-vs-superseded
 * / free-text q) and the limit/offset window live in the URL search params so a
 * filtered view is shareable and survives reload / the browser back button. The
 * top-bar global search writes `?q=` and the top-bar scope is layered in from
 * ScopeContext (useScope) — this hook merges all of that into the typed MemoriesQuery
 * the useMemories hook consumes verbatim.
 *
 * Page-local to the Memories screen.
 */

import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import type { MemoriesQuery, MemoryType, Tier } from "@/api/types";
import { MEMORY_TYPES, TIERS } from "@/api/types";

export const PAGE_SIZE = 50;

/** Tri-state active filter as a URL-friendly string. */
export type ActiveFilter = "all" | "active" | "superseded";

export interface MemoriesFilters {
  type: MemoryType | "";
  tier: Tier | "";
  entity: string;
  source: string;
  active: ActiveFilter;
  q: string;
  offset: number;
}

function readType(v: string | null): MemoryType | "" {
  return v && (MEMORY_TYPES as readonly string[]).includes(v) ? (v as MemoryType) : "";
}
function readTier(v: string | null): Tier | "" {
  return v && (TIERS as readonly string[]).includes(v) ? (v as Tier) : "";
}
function readActive(v: string | null): ActiveFilter {
  return v === "active" || v === "superseded" ? v : "all";
}

export interface UseMemoriesFiltersResult {
  filters: MemoriesFilters;
  /** The typed query for useMemories (top-bar scope merged in). */
  query: MemoriesQuery;
  /** Patch one or more filters; any filter change (except offset) resets paging. */
  setFilters: (patch: Partial<MemoriesFilters>) => void;
  clearAll: () => void;
  setOffset: (offset: number) => void;
  /** True when any user-controlled filter is non-default. */
  hasActiveFilters: boolean;
  page: { limit: number; offset: number };
}

export function useMemoriesFilters(scope: string | null): UseMemoriesFiltersResult {
  const [params, setParams] = useSearchParams();

  const filters = useMemo<MemoriesFilters>(() => {
    const offsetRaw = Number(params.get("offset"));
    return {
      type: readType(params.get("type")),
      tier: readTier(params.get("tier")),
      entity: params.get("entity") ?? "",
      source: params.get("source") ?? "",
      active: readActive(params.get("active")),
      q: params.get("q") ?? "",
      offset: Number.isFinite(offsetRaw) && offsetRaw > 0 ? Math.floor(offsetRaw) : 0,
    };
  }, [params]);

  const setFilters = useCallback(
    (patch: Partial<MemoriesFilters>) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          const apply = (key: string, value: string | undefined) => {
            if (value === undefined || value === "") next.delete(key);
            else next.set(key, value);
          };
          if ("type" in patch) apply("type", patch.type);
          if ("tier" in patch) apply("tier", patch.tier);
          if ("entity" in patch) apply("entity", patch.entity);
          if ("source" in patch) apply("source", patch.source);
          if ("active" in patch) apply("active", patch.active === "all" ? "" : patch.active);
          if ("q" in patch) apply("q", patch.q);
          if ("offset" in patch) {
            apply("offset", patch.offset && patch.offset > 0 ? String(patch.offset) : undefined);
          } else {
            // Any filter change other than an explicit offset resets to page 1.
            next.delete("offset");
          }
          return next;
        },
        { replace: true },
      );
    },
    [setParams],
  );

  const setOffset = useCallback((offset: number) => setFilters({ offset }), [setFilters]);

  const clearAll = useCallback(() => {
    setParams(() => new URLSearchParams(), { replace: true });
  }, [setParams]);

  const query = useMemo<MemoriesQuery>(() => {
    const q: MemoriesQuery = { limit: PAGE_SIZE, offset: filters.offset };
    if (filters.type) q.type = filters.type;
    if (filters.tier) q.tier = filters.tier;
    if (filters.entity.trim()) q.entity = filters.entity.trim();
    if (filters.source.trim()) q.source = filters.source.trim();
    if (filters.q.trim()) q.q = filters.q.trim();
    if (filters.active === "active") q.active = true;
    else if (filters.active === "superseded") q.active = false;
    // Top-bar scope (useScope) layers over the URL filters; null = all scopes.
    if (scope) q.scope = scope;
    return q;
  }, [filters, scope]);

  const hasActiveFilters =
    filters.type !== "" ||
    filters.tier !== "" ||
    filters.entity.trim() !== "" ||
    filters.source.trim() !== "" ||
    filters.active !== "all" ||
    filters.q.trim() !== "";

  return {
    filters,
    query,
    setFilters,
    clearAll,
    setOffset,
    hasActiveFilters,
    page: { limit: PAGE_SIZE, offset: filters.offset },
  };
}
