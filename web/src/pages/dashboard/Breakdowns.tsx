/**
 * Dashboard breakdown panels (PRD §4.1): totals-by-type, hot-vs-archive tier
 * split, and source breakdown (claude_code / openai / hermes / …). Page-local to
 * the Dashboard; each reads from the StoreStatsResponse the page already fetched
 * and renders proportion bars via the local BreakdownRow widget + shared badges.
 */

import { Panel } from "@/components/primitives";
import { TypeBadge, TierBadge, Badge } from "@/components/Badge";
import type { MemoryType, StoreStatsResponse, Tier } from "@/api/types";
import { MEMORY_TYPES, TIERS } from "@/api/types";
import { TYPE_BADGE, TIER_BADGE } from "@/theme/tokens";
import { BreakdownRow, MutedHint } from "@/pages/dashboard/widgets";

/** Totals broken down by memory type (preference / project_fact / idea_seed). */
export function TotalsByType({ stats }: { stats: StoreStatsResponse | undefined }) {
  const byType = stats?.by_type ?? {};
  const total = stats?.total_memories ?? Object.values(byType).reduce((a, b) => a + b, 0);
  // Known types first, in canonical order; then any unexpected keys the backend sends.
  const extraKeys = Object.keys(byType).filter((k) => !MEMORY_TYPES.includes(k as MemoryType));

  return (
    <Panel title="By type">
      <div className="flex flex-col gap-2.5">
        {MEMORY_TYPES.map((t) => (
          <BreakdownRow
            key={t}
            label={t}
            labelNode={<TypeBadge type={t} />}
            count={byType[t] ?? 0}
            total={total}
            fillClass={TYPE_BADGE[t].dot}
          />
        ))}
        {extraKeys.map((k) => (
          <BreakdownRow key={k} label={k} count={byType[k] ?? 0} total={total} fillClass="bg-text-faint" />
        ))}
      </div>
    </Panel>
  );
}

/** Hot vs archive tier split. */
export function TierSplit({ stats }: { stats: StoreStatsResponse | undefined }) {
  const byTier = stats?.by_tier ?? {};
  const total = Object.values(byTier).reduce((a, b) => a + b, 0) || (stats?.total_memories ?? 0);
  const extraKeys = Object.keys(byTier).filter((k) => !TIERS.includes(k as Tier));

  return (
    <Panel title="By tier">
      <div className="flex flex-col gap-2.5">
        {TIERS.map((t) => (
          <BreakdownRow
            key={t}
            label={t}
            labelNode={<TierBadge tier={t} />}
            count={byTier[t] ?? 0}
            total={total}
            fillClass={TIER_BADGE[t].dot}
          />
        ))}
        {extraKeys.map((k) => (
          <BreakdownRow key={k} label={k} count={byTier[k] ?? 0} total={total} fillClass="bg-text-faint" />
        ))}
      </div>
    </Panel>
  );
}

/** Source breakdown — which ingestion source produced the memories. */
export function SourceBreakdown({ stats }: { stats: StoreStatsResponse | undefined }) {
  const bySource = stats?.by_source ?? {};
  const entries = Object.entries(bySource).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((a, [, v]) => a + v, 0);

  return (
    <Panel title="By source">
      {entries.length === 0 ? (
        <MutedHint>no source data</MutedHint>
      ) : (
        <div className="flex flex-col gap-2.5">
          {entries.map(([source, count]) => (
            <BreakdownRow
              key={source}
              label={source}
              labelNode={
                <Badge outline className="normal-case">
                  {source}
                </Badge>
              }
              count={count}
              total={total}
              fillClass="bg-accent"
            />
          ))}
        </div>
      )}
    </Panel>
  );
}
