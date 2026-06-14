/**
 * Dashboard breakdown panels (PRD §4.1): totals-by-CATEGORY (free-form, discovered),
 * by-SCOPE-decision (global vs project), hot-vs-archive tier split, and source
 * breakdown (claude_code / openai / hermes / …). Page-local to the Dashboard; each
 * reads from the StoreStatsResponse the page already fetched and renders proportion
 * bars via the local BreakdownRow widget + shared badges.
 *
 * The core data-model redesign replaced the fixed by_type panel with the open-ended
 * by_category panel (any emergent category, sorted by count, colored deterministically)
 * and added a by_scope_decision panel.
 */

import { Panel } from "@/components/primitives";
import { CategoryBadge, TierBadge, Badge } from "@/components/Badge";
import type { StoreStatsResponse, Tier } from "@/api/types";
import { TIERS } from "@/api/types";
import { TIER_BADGE, categoryColor } from "@/theme/tokens";
import { BreakdownRow, MutedHint } from "@/pages/dashboard/widgets";

/** Totals broken down by FREE-FORM category (discovered, sorted by count). */
export function TotalsByCategory({ stats }: { stats: StoreStatsResponse | undefined }) {
  const byCategory = stats?.by_category ?? {};
  const entries = Object.entries(byCategory).sort((a, b) => b[1] - a[1]);
  const total =
    stats?.total_memories ?? entries.reduce((acc, [, v]) => acc + v, 0);

  return (
    <Panel title="By category">
      {entries.length === 0 ? (
        <MutedHint>no category data</MutedHint>
      ) : (
        <div className="flex flex-col gap-2.5">
          {entries.map(([category, count]) => (
            <BreakdownRow
              key={category}
              label={category}
              labelNode={<CategoryBadge category={category} />}
              count={count}
              total={total}
              fillColor={categoryColor(category).fg}
            />
          ))}
        </div>
      )}
    </Panel>
  );
}

/** Totals broken down by the controlled scope decision (global vs project). */
export function ScopeDecisionBreakdown({ stats }: { stats: StoreStatsResponse | undefined }) {
  const byDecision = stats?.by_scope_decision ?? {};
  const total = Object.values(byDecision).reduce((a, b) => a + b, 0) || (stats?.total_memories ?? 0);
  const order: Array<{ key: string; fillClass: string }> = [
    { key: "global", fillClass: "bg-info" },
    { key: "project", fillClass: "bg-accent" },
  ];
  const known = new Set(order.map((o) => o.key));
  const extraKeys = Object.keys(byDecision).filter((k) => !known.has(k));

  return (
    <Panel title="By scope">
      <div className="flex flex-col gap-2.5">
        {order.map(({ key, fillClass }) => (
          <BreakdownRow
            key={key}
            label={key}
            count={byDecision[key] ?? 0}
            total={total}
            fillClass={fillClass}
          />
        ))}
        {extraKeys.map((k) => (
          <BreakdownRow key={k} label={k} count={byDecision[k] ?? 0} total={total} fillClass="bg-text-faint" />
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
