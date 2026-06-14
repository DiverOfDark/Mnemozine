/**
 * IndexPreview (FE-B / PRD §4.5, FR-RET-3) — a live preview of the ~500-token
 * SessionStart injection index that recall would emit. Shows a token-budget meter
 * (estimate / budget, turning warn→danger as it approaches/exceeds budget), the
 * preference / project_fact counts, the idea-seed hints + entity tags, and the raw
 * index text in a mono panel (the actual bytes that would be injected).
 *
 * This is the precision-debugging payload — the operator reads exactly what the
 * model would see at SessionStart.
 */

import { Badge } from "@/components/index";
import type { InjectionIndexPreview } from "@/api";
import { cn } from "@/lib/cn";
import { TYPE_BADGE } from "@/theme/tokens";

export function IndexPreview({ preview }: { preview: InjectionIndexPreview }) {
  const { token_estimate, token_budget } = preview;
  const pct = token_budget > 0 ? token_estimate / token_budget : 0;
  const over = token_estimate > token_budget;
  const near = !over && pct >= 0.85;
  const meterColor = over ? "bg-danger" : near ? "bg-warn" : "bg-tier-hot";
  const meterText = over ? "text-danger" : near ? "text-warn" : "text-tier-hot";

  return (
    <div className="flex flex-col gap-3">
      {/* token budget meter ------------------------------------------------- */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-baseline justify-between">
          <span className="text-2xs uppercase tracking-wide text-text-faint">token budget</span>
          <span className={cn("font-mono text-2xs tabular-nums", meterText)}>
            {token_estimate.toLocaleString()} / {token_budget.toLocaleString()}
            {over && " · OVER BUDGET"}
          </span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-bg-inset">
          <div
            className={cn("h-full rounded-full transition-all duration-fast", meterColor)}
            style={{ width: `${Math.min(100, Math.max(2, pct * 100))}%` }}
          />
        </div>
      </div>

      {/* counts ------------------------------------------------------------- */}
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge textClass={TYPE_BADGE.preference.text} bgClass={TYPE_BADGE.preference.bg} dotClass={TYPE_BADGE.preference.dot}>
          {preview.preference_count} preferences
        </Badge>
        <Badge
          textClass={TYPE_BADGE.project_fact.text}
          bgClass={TYPE_BADGE.project_fact.bg}
          dotClass={TYPE_BADGE.project_fact.dot}
        >
          {preview.project_fact_count} project_facts
        </Badge>
      </div>

      {/* idea-seed hints ---------------------------------------------------- */}
      {preview.idea_seed_hints.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <span className="text-2xs uppercase tracking-wide text-text-faint">idea-seed hints</span>
          <div className="flex flex-wrap gap-1.5">
            {preview.idea_seed_hints.map((hint, i) => (
              <Badge
                key={i}
                textClass={TYPE_BADGE.idea_seed.text}
                bgClass={TYPE_BADGE.idea_seed.bg}
                dotClass={TYPE_BADGE.idea_seed.dot}
                className="normal-case"
              >
                {hint}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* entity tags -------------------------------------------------------- */}
      {preview.entity_tags.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <span className="text-2xs uppercase tracking-wide text-text-faint">entity tags</span>
          <div className="flex flex-wrap gap-1.5">
            {preview.entity_tags.map((tag) => (
              <Badge key={tag} outline textClass="text-text-muted" className="normal-case">
                {tag}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* the raw injected index --------------------------------------------- */}
      <div className="flex flex-col gap-1.5">
        <span className="text-2xs uppercase tracking-wide text-text-faint">
          injected index (verbatim)
        </span>
        <pre className="max-h-80 overflow-auto rounded-md border border-border bg-bg p-3 font-mono text-xs leading-relaxed text-text-muted">
          <code className="whitespace-pre-wrap break-words">{preview.text}</code>
        </pre>
      </div>
    </div>
  );
}
