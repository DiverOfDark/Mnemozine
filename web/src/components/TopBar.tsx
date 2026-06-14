/**
 * TopBar — the global top bar (PRD §4 wireframe): global search + scope filter +
 * live store stats + infra health dots. Reads useStats / useHealth so the counts
 * are live everywhere. The search box navigates to /memories?q=… ; the scope
 * filter is surfaced via the ScopeContext so any screen can read the active scope.
 *
 * App chrome — screen agents don't edit this; they consume `useScope()` instead.
 */

import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { useHealth, useStats } from "@/api/hooks";
import { useScope } from "@/state/scope";
import { HEALTH_STATUS } from "@/theme/tokens";
import { hasToken } from "@/api/client";
import { Input } from "@/components/primitives";
import { cn } from "@/lib/cn";

function StatPill({ label, value, tone }: { label: string; value: number | string; tone?: string }) {
  return (
    <div className="flex items-baseline gap-1.5 whitespace-nowrap">
      <span className="text-2xs uppercase tracking-wide text-text-faint">{label}</span>
      <span className={cn("font-mono text-xs tabular-nums", tone ?? "text-text")}>{value}</span>
    </div>
  );
}

export function TopBar() {
  const navigate = useNavigate();
  const { scope, setScope } = useScope();
  const [search, setSearch] = useState("");

  const { data: stats } = useStats();
  const { data: health } = useHealth({ refetchInterval: 30_000 });

  const onSearch = (e: FormEvent) => {
    e.preventDefault();
    const q = search.trim();
    navigate(q ? `/memories?q=${encodeURIComponent(q)}` : "/memories");
  };

  return (
    <header className="flex h-topbar shrink-0 items-center gap-4 border-b border-border bg-bg-raised px-4">
      {/* global search */}
      <form onSubmit={onSearch} className="flex w-72 items-center">
        <div className="relative w-full">
          <svg
            className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-text-faint"
            width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3"
          >
            <circle cx="7" cy="7" r="4.5" />
            <path d="M11 11l3 3" strokeLinecap="round" />
          </svg>
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search memories…"
            className="h-7 w-full pl-7"
            aria-label="Global search"
          />
        </div>
      </form>

      {/* scope filter */}
      <div className="flex items-center gap-1.5">
        <span className="text-2xs uppercase tracking-wide text-text-faint">scope</span>
        <input
          value={scope ?? ""}
          onChange={(e) => setScope(e.target.value || null)}
          placeholder="all"
          list="scope-suggestions"
          aria-label="Scope filter"
          className="h-7 w-36 rounded border border-border-strong bg-bg px-2 font-mono text-xs text-text placeholder:text-text-faint focus:border-border-focus"
        />
        <datalist id="scope-suggestions">
          <option value="global" />
        </datalist>
      </div>

      <div className="h-5 w-px bg-border" />

      {/* live store stats */}
      <div className="flex items-center gap-4 overflow-x-auto">
        <StatPill label="mem" value={stats?.total_memories ?? "—"} />
        <StatPill label="active" value={stats?.active_count ?? "—"} tone="text-active" />
        <StatPill label="superseded" value={stats?.superseded_count ?? "—"} tone="text-superseded" />
        <StatPill label="entities" value={stats?.entity_count ?? "—"} />
      </div>

      <div className="ml-auto flex items-center gap-3">
        {/* infra health dots */}
        <div className="flex items-center gap-2.5">
          {(health?.components ?? []).map((c) => {
            const s = HEALTH_STATUS[c.status] ?? HEALTH_STATUS.unknown!;
            return (
              <span
                key={c.name}
                title={`${c.name}: ${c.status}${c.detail ? ` — ${c.detail}` : ""}`}
                className="flex items-center gap-1.5 text-2xs text-text-muted"
              >
                <span className={cn("h-1.5 w-1.5 rounded-full", s.dot)} />
                <span className="font-mono">{c.name}</span>
              </span>
            );
          })}
        </div>

        {/* lock indicator when a token is configured */}
        {hasToken() && (
          <span title="bearer token active" className="text-text-faint">
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3">
              <rect x="3" y="7" width="10" height="6.5" rx="1" />
              <path d="M5 7V5a3 3 0 016 0v2" />
            </svg>
          </span>
        )}
      </div>
    </header>
  );
}
