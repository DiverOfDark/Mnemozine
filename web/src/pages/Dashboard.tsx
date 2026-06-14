/**
 * Dashboard (PRD §4.1) — the operator's at-a-glance console home: totals by memory
 * type, hot vs archive, a store-growth sparkline, source breakdown, the recent
 * activity feed, maintenance job status, and infra health tiles (FalkorDB / Ollama
 * / LLM endpoint).
 *
 * Composes the shared design-system components + the typed api hooks (useStats,
 * useHealth, useActivity, useMaintenance) consumed via page-local widgets under
 * src/pages/dashboard/. Honors the dark observability theme and color-by-type/tier.
 *
 * Owns ONLY this file + src/pages/dashboard/**. Does not touch the router, the api
 * client/hooks, the theme, or the shared components.
 */

import { Link } from "react-router-dom";
import { Page } from "@/components/AppShell";
import { Loading, ErrorState } from "@/components/primitives";
import { useStats } from "@/api/hooks";
import { useScope } from "@/state/scope";
import { parseScope, pluralize } from "@/lib/format";
import { PATHS } from "@/routes";

import { StatTile } from "@/pages/dashboard/widgets";
import { GrowthPanel } from "@/pages/dashboard/GrowthPanel";
import { TotalsByType, TierSplit, SourceBreakdown } from "@/pages/dashboard/Breakdowns";
import { ActivityFeed } from "@/pages/dashboard/ActivityFeed";
import { MaintenanceStatus } from "@/pages/dashboard/MaintenanceStatus";
import { HealthTiles } from "@/pages/dashboard/HealthTiles";

export default function Dashboard() {
  const { scope } = useScope();
  const scopeLabel = scope ? parseScope(scope).label : "all scopes";
  const { data: stats, isLoading, error, refetch } = useStats({ refetchInterval: 30_000 });

  const activePct =
    stats && stats.total_memories > 0
      ? Math.round((stats.active_count / stats.total_memories) * 100)
      : null;

  return (
    <Page
      title="Dashboard"
      subtitle={
        <span>
          memory layer overview · scope <span className="font-mono text-text-muted">{scopeLabel}</span>
        </span>
      }
      bodyClassName="flex flex-col gap-4"
    >
      {/* Totals strip */}
      {isLoading ? (
        <Loading label="Loading store stats…" />
      ) : error ? (
        <ErrorState error={error} onRetry={() => void refetch()} />
      ) : (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          <Link to={PATHS.memories} className="contents">
            <StatTile
              label="memories"
              value={(stats?.total_memories ?? 0).toLocaleString()}
              sub={pluralize(stats?.entity_count ?? 0, "entity", "entities")}
            />
          </Link>
          <StatTile
            label="active"
            value={(stats?.active_count ?? 0).toLocaleString()}
            tone="text-active"
            dotClass="bg-active"
            sub={activePct != null ? `${activePct}% of store` : undefined}
          />
          <StatTile
            label="superseded"
            value={(stats?.superseded_count ?? 0).toLocaleString()}
            tone="text-superseded"
            dotClass="bg-superseded"
          />
          <StatTile
            label="hot"
            value={(stats?.by_tier?.["hot"] ?? 0).toLocaleString()}
            tone="text-tier-hot"
            dotClass="bg-tier-hot"
          />
          <StatTile
            label="archive"
            value={(stats?.by_tier?.["archive"] ?? 0).toLocaleString()}
            tone="text-tier-archive"
            dotClass="bg-tier-archive"
          />
        </div>
      )}

      {/* Growth + breakdowns */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <GrowthPanel scope={scope} />
        <TotalsByType stats={stats} />
        <TierSplit stats={stats} />
        <SourceBreakdown stats={stats} />
      </div>

      {/* Activity + maintenance + health */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <ActivityFeed scope={scope} />
        </div>
        <div className="flex flex-col gap-4 lg:col-span-2">
          <MaintenanceStatus />
          <HealthTiles />
        </div>
      </div>
    </Page>
  );
}
