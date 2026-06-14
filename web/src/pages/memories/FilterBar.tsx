/**
 * FilterBar — the Memories table's filter toolbar (PRD §4.2). Drives the URL-backed
 * filters from useMemoriesFilters: category, tier, active-vs-superseded, entity,
 * source, and free-text (q). The top-bar scope filter (ScopeContext) is shown
 * read-only here since it is owned by the global TopBar. Text filters are debounced
 * so typing doesn't refetch on every keystroke. Page-local to the Memories screen.
 *
 * The old fixed `type` enum dropdown is now a DYNAMIC CATEGORY facet: it lists the
 * categories actually discovered in the store (with counts) from the
 * /memories/facets/categories endpoint — categories are open-ended, so there is no
 * hard-coded option list.
 */

import { useEffect, useRef, useState } from "react";
import { Field, Select, Input, Button } from "@/components/primitives";
import { TIERS, type CategoryFacet } from "@/api/types";
import type { ActiveFilter, MemoriesFilters } from "@/pages/memories/useMemoriesFilters";
import { parseScope } from "@/lib/format";

/** A text input that commits to the URL filters after a debounce / on Enter / on blur. */
function DebouncedField({
  label,
  value,
  placeholder,
  onCommit,
  widthClass = "w-36",
  mono = false,
}: {
  label: string;
  value: string;
  placeholder?: string;
  onCommit: (value: string) => void;
  widthClass?: string;
  mono?: boolean;
}) {
  const [draft, setDraft] = useState(value);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep the draft in sync when the URL changes from outside (e.g. clear-all, back).
  useEffect(() => setDraft(value), [value]);

  const schedule = (next: string) => {
    setDraft(next);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => onCommit(next), 350);
  };
  const commitNow = () => {
    if (timer.current) clearTimeout(timer.current);
    if (draft !== value) onCommit(draft);
  };

  useEffect(() => () => void (timer.current && clearTimeout(timer.current)), []);

  return (
    <Field label={label}>
      <Input
        value={draft}
        placeholder={placeholder}
        onChange={(e) => schedule(e.target.value)}
        onBlur={commitNow}
        onKeyDown={(e) => {
          if (e.key === "Enter") commitNow();
          if (e.key === "Escape") {
            setDraft(value);
            (e.target as HTMLInputElement).blur();
          }
        }}
        className={mono ? `${widthClass} font-mono` : widthClass}
      />
    </Field>
  );
}

interface FilterBarProps {
  filters: MemoriesFilters;
  scope: string | null;
  setFilters: (patch: Partial<MemoriesFilters>) => void;
  clearAll: () => void;
  hasActiveFilters: boolean;
  /** Total rows matching the current filters (for the result count chip). */
  total?: number;
  /** Discovered category facets (category + count) for the dynamic CATEGORY filter. */
  categoryFacets?: CategoryFacet[];
}

export function FilterBar({
  filters,
  scope,
  setFilters,
  clearAll,
  hasActiveFilters,
  total,
  categoryFacets = [],
}: FilterBarProps) {
  const scopeLabel = scope ? parseScope(scope).label : "all";
  // If the active category isn't in the discovered set (e.g. a stale URL filter or
  // a just-emptied store), still show it as an option so the value round-trips.
  const knownCategories = categoryFacets.map((f) => f.category);
  const showActiveAsExtra =
    filters.category !== "" && !knownCategories.includes(filters.category);

  return (
    <div className="flex flex-wrap items-end gap-3">
      <DebouncedField
        label="search"
        value={filters.q}
        placeholder="content / id…"
        onCommit={(q) => setFilters({ q })}
        widthClass="w-48"
      />

      <Field label="category">
        <Select
          value={filters.category}
          onChange={(e) => setFilters({ category: e.target.value })}
          title="Filter by discovered free-form category"
        >
          <option value="">all categories</option>
          {categoryFacets.map((f) => (
            <option key={f.category} value={f.category}>
              {f.category} ({f.count})
            </option>
          ))}
          {showActiveAsExtra && (
            <option value={filters.category}>{filters.category}</option>
          )}
        </Select>
      </Field>

      <Field label="tier">
        <Select value={filters.tier} onChange={(e) => setFilters({ tier: e.target.value as MemoriesFilters["tier"] })}>
          <option value="">all</option>
          {TIERS.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </Select>
      </Field>

      <Field label="state">
        <Select value={filters.active} onChange={(e) => setFilters({ active: e.target.value as ActiveFilter })}>
          <option value="all">all</option>
          <option value="active">active</option>
          <option value="superseded">superseded</option>
        </Select>
      </Field>

      <DebouncedField
        label="entity"
        value={filters.entity}
        placeholder="entity name…"
        onCommit={(entity) => setFilters({ entity })}
        mono
      />

      <DebouncedField
        label="source"
        value={filters.source}
        placeholder="claude_code…"
        onCommit={(source) => setFilters({ source })}
        mono
      />

      <Field label="scope (top bar)">
        <span
          className="flex h-7 items-center rounded border border-border bg-bg-inset px-2 font-mono text-xs text-text-muted"
          title="Scope is set from the global top-bar scope filter"
        >
          {scopeLabel}
        </span>
      </Field>

      <div className="ml-auto flex items-center gap-2 pb-0.5">
        {typeof total === "number" && (
          <span className="font-mono text-2xs text-text-faint tabular-nums">
            {total.toLocaleString()} match{total === 1 ? "" : "es"}
          </span>
        )}
        {hasActiveFilters && (
          <Button variant="ghost" onClick={clearAll}>
            clear filters
          </Button>
        )}
      </div>
    </div>
  );
}
