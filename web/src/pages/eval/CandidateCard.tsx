/**
 * CandidateCard — a single F4 bootstrap-labeling candidate (PRD §4.8).
 *
 * Shows the candidate's proposed type, content, scope, entities, confidence, and
 * source session, plus the four label choices (preference / project_fact / idea_seed
 * / not-a-memory) as buttons with their keyboard shortcut hints. The currently-applied
 * choice (derived from the persisted {label, corrected_type}) is highlighted; a dropped
 * ("not-a-memory") candidate is greyed via the shared `.superseded` treatment. The
 * focused card (keyboard cursor) gets an accent ring so the operator always knows which
 * candidate the 1/2/3/0 keys will label.
 *
 * Page-local: design-system components + the page-local label mapping only.
 */

import { forwardRef } from "react";

import type { BootstrapCandidate } from "@/api";
import { Badge, Kbd, ScoreBar, TypeBadge } from "@/components";
import { cn } from "@/lib/cn";
import { shortId, parseScope } from "@/lib/format";
import { CHOICES, choiceFromCandidate, type LabelChoice } from "@/pages/eval/bootstrapLabels";

interface CandidateCardProps {
  candidate: BootstrapCandidate;
  focused: boolean;
  saving: boolean;
  onChoose: (choice: LabelChoice) => void;
  onFocus: () => void;
}

export const CandidateCard = forwardRef<HTMLDivElement, CandidateCardProps>(function CandidateCard(
  { candidate, focused, saving, onChoose, onFocus },
  ref,
) {
  const applied = choiceFromCandidate(candidate);
  const dropped = applied === "not_a_memory";
  const scope = parseScope(candidate.scope);

  return (
    <div
      ref={ref}
      tabIndex={0}
      onMouseEnter={onFocus}
      onFocus={onFocus}
      className={cn(
        "flex flex-col gap-3 rounded-md border bg-bg-raised p-3 outline-none transition-colors",
        focused ? "border-accent ring-1 ring-accent" : "border-border hover:border-border-strong",
        dropped && "opacity-60",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <TypeBadge type={candidate.proposed_type} />
          <span className="text-2xs uppercase tracking-wide text-text-faint">proposed</span>
          {applied && <AppliedChip choice={applied} proposed={candidate.proposed_type === applied} />}
        </div>
        <div className="flex items-center gap-2">
          <ScoreBar value={candidate.confidence} format="percent" width={56} />
          {saving && <span className="text-2xs text-text-faint">saving…</span>}
        </div>
      </div>

      <p className={cn("text-sm leading-relaxed", dropped ? "superseded" : "text-text")}>{candidate.content}</p>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-2xs text-text-muted">
        <span>
          <span className="text-text-faint">scope</span>{" "}
          <code className="font-mono text-text">{scope.label}</code>
        </span>
        {candidate.entities.length > 0 && (
          <span className="flex flex-wrap items-center gap-1">
            <span className="text-text-faint">entities</span>
            {candidate.entities.slice(0, 6).map((e) => (
              <Badge key={e} textClass="text-text-muted" bgClass="bg-bg-inset">
                {e}
              </Badge>
            ))}
            {candidate.entities.length > 6 && (
              <span className="text-text-faint">+{candidate.entities.length - 6}</span>
            )}
          </span>
        )}
        <span>
          <span className="text-text-faint">session</span>{" "}
          <code className="font-mono text-text-faint">{shortId(candidate.source_session, 10)}</code>
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-1.5 border-t border-border pt-2.5">
        {CHOICES.map((c) => {
          const active = applied === c.choice;
          return (
            <button
              key={c.choice}
              type="button"
              disabled={saving}
              onClick={() => onChoose(c.choice)}
              className={cn(
                "inline-flex items-center gap-1.5 rounded border px-2 py-1 text-xs font-medium transition-colors duration-fast disabled:cursor-not-allowed disabled:opacity-50",
                active
                  ? cn(c.textClass, c.bgClass, "border-current")
                  : "border-border-strong bg-bg-inset text-text-muted hover:text-text hover:bg-bg-hover",
              )}
            >
              <Kbd>{c.key}</Kbd>
              {c.label}
            </button>
          );
        })}
      </div>
    </div>
  );
});

function AppliedChip({ choice, proposed }: { choice: LabelChoice; proposed: boolean }) {
  if (choice === "not_a_memory") {
    return (
      <Badge textClass="text-danger" bgClass="bg-danger/10" dotClass="bg-danger">
        not-a-memory
      </Badge>
    );
  }
  return (
    <span className="flex items-center gap-1">
      <span className="text-text-faint">→</span>
      <TypeBadge type={choice} />
      {!proposed && <span className="text-2xs text-warn" title="reclassified from the proposed type">reclassified</span>}
    </span>
  );
}
