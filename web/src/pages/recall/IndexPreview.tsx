/**
 * IndexPreview (FE-B / PRD §4.5, FR-RET-3) — a live preview of the ~500-token
 * SessionStart injection index that recall would emit. Shows a token-budget meter
 * (estimate / budget, turning warn→danger as it approaches/exceeds budget), the
 * global / project SCOPE counts, the cross-reference hints + entity tags, and the
 * raw index text in a mono panel (the actual bytes that would be injected).
 *
 * The core data-model redesign renamed the counts: preference→global,
 * project_fact→project, idea_seed_hints→cross_ref_hints (the index now reports by
 * the controlled scope decision, not the old type enum).
 *
 * This is the precision-debugging payload — the operator reads exactly what the
 * model would see at SessionStart.
 */

import { Badge } from "@/components/index";
import type { InjectionIndexPreview } from "@/api";
import { cn } from "@/lib/cn";

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

      {/* scope counts ------------------------------------------------------- */}
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge textClass="text-info" bgClass="bg-bg-inset" dotClass="bg-info">
          {preview.global_count} global
        </Badge>
        <Badge textClass="text-accent" bgClass="bg-accent-muted" dotClass="bg-accent">
          {preview.project_count} project
        </Badge>
      </div>

      {/* cross-reference hints ---------------------------------------------- */}
      {preview.cross_ref_hints.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <span className="text-2xs uppercase tracking-wide text-text-faint">
            cross-reference hints
          </span>
          <div className="flex flex-wrap gap-1.5">
            {preview.cross_ref_hints.map((hint, i) => (
              <Badge
                key={i}
                outline
                style={{ color: "#fb7185" }}
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
