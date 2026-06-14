/**
 * BootstrapLabeler — the F4 browser bootstrap-labeling workflow (PRD §4.8).
 *
 * Presents the auto-proposed candidates (useBootstrapCandidates) as a keyboard-first
 * card stack. The operator labels each candidate preference / project_fact / idea_seed
 * / not-a-memory; labels persist through useLabelBootstrap (which maps the four choices
 * onto the wire {label, corrected_type} pair). useFinishBootstrap folds the kept
 * candidates into the gold set and returns the refreshed EvalSummaryResponse.
 *
 * Keyboard affordances (PRD §5):
 *   j / ↓   next candidate          k / ↑   previous candidate
 *   1 / 2 / 3  label preference / project_fact / idea_seed  (auto-advances)
 *   0 / n   not-a-memory (drop)      u   clear → unreviewed
 *   f       finish & save the gold set
 *
 * Page-local: design-system components + the typed api hooks + page-local helpers.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  useBootstrapCandidates,
  useFinishBootstrap,
  useLabelBootstrap,
  type BootstrapCandidate,
} from "@/api";
import { Button, EmptyState, ErrorState, KeyboardHints, Loading, Panel, type KeyHint } from "@/components";
import { CandidateCard } from "@/pages/eval/CandidateCard";
import {
  UNREVIEWED_BODY,
  bodyForChoice,
  choiceForKey,
  choiceFromCandidate,
  type LabelChoice,
} from "@/pages/eval/bootstrapLabels";

const HINTS: KeyHint[] = [
  { keys: ["j"], label: "next" },
  { keys: ["k"], label: "prev" },
  { keys: ["1"], label: "preference" },
  { keys: ["2"], label: "project_fact" },
  { keys: ["3"], label: "idea_seed" },
  { keys: ["0"], label: "not-a-memory" },
  { keys: ["u"], label: "unreviewed" },
  { keys: ["f"], label: "finish" },
];

export function BootstrapLabeler() {
  const { data, isLoading, error, refetch } = useBootstrapCandidates();
  const label = useLabelBootstrap();
  const finish = useFinishBootstrap();

  const candidates = useMemo<BootstrapCandidate[]>(() => data?.candidates ?? [], [data]);
  const [cursor, setCursor] = useState(0);
  const cardRefs = useRef<(HTMLDivElement | null)[]>([]);

  // Keep the cursor within bounds when the candidate list changes (invalidation).
  useEffect(() => {
    setCursor((c) => Math.max(0, Math.min(candidates.length - 1, c)));
  }, [candidates.length]);

  const move = useCallback(
    (delta: number) => {
      setCursor((c) => {
        const next = Math.max(0, Math.min(candidates.length - 1, c + delta));
        cardRefs.current[next]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
        return next;
      });
    },
    [candidates.length],
  );

  const applyChoice = useCallback(
    (index: number, choice: LabelChoice, advance: boolean) => {
      const candidate = candidates[index];
      if (!candidate) return;
      label.mutate({ candidateId: candidate.candidate_id, body: bodyForChoice(choice) });
      if (advance) move(1);
    },
    [candidates, label, move],
  );

  const clearChoice = useCallback(
    (index: number) => {
      const candidate = candidates[index];
      if (!candidate) return;
      label.mutate({ candidateId: candidate.candidate_id, body: UNREVIEWED_BODY });
    },
    [candidates, label],
  );

  // Global keyboard handler for the labeler shortcuts.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.matches("input, textarea, select, [contenteditable='true']")) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      switch (e.key) {
        case "j":
        case "ArrowDown":
          e.preventDefault();
          move(1);
          return;
        case "k":
        case "ArrowUp":
          e.preventDefault();
          move(-1);
          return;
        case "u":
          e.preventDefault();
          clearChoice(cursor);
          return;
        case "f":
          e.preventDefault();
          if (!finish.isPending) finish.mutate();
          return;
        default: {
          const choice = choiceForKey(e.key);
          if (choice) {
            e.preventDefault();
            applyChoice(cursor, choice, true);
          }
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cursor, move, clearChoice, applyChoice, finish]);

  const reviewedCount = candidates.filter((c) => choiceFromCandidate(c) !== null).length;
  const keptCount = candidates.filter((c) => {
    const ch = choiceFromCandidate(c);
    return ch !== null && ch !== "not_a_memory";
  }).length;
  const savingId = label.isPending ? label.variables?.candidateId : undefined;

  return (
    <Panel
      title="Bootstrap labeling — F4"
      bodyClassName="flex flex-col gap-3 p-0"
      actions={
        <div className="flex items-center gap-3">
          <span className="font-mono text-2xs text-text-muted tabular-nums">
            {reviewedCount}/{candidates.length} reviewed · {keptCount} kept
          </span>
          <Button
            variant="primary"
            loading={finish.isPending}
            disabled={candidates.length === 0}
            onClick={() => finish.mutate()}
            title="fold the kept candidates into the gold set"
          >
            finish &amp; save gold set
          </Button>
        </div>
      }
    >
      <div className="border-b border-border bg-bg-inset px-3 py-2">
        <ProgressBar reviewed={reviewedCount} total={candidates.length} />
        <KeyboardHints hints={HINTS} className="mt-2" />
      </div>

      {finish.isError && (
        <div className="mx-3 rounded border border-danger/40 bg-danger/5 px-3 py-1.5 text-2xs text-danger">
          finish failed: {finish.error instanceof Error ? finish.error.message : String(finish.error)}
        </div>
      )}
      {finish.isSuccess && (
        <div className="mx-3 rounded border border-ok/40 bg-tier-bg-hot px-3 py-1.5 text-2xs text-ok">
          gold set saved — {finish.data.gold_set} now {finish.data.passed ? "passing" : "failing"} (
          {finish.data.metrics.length} metrics).
        </div>
      )}
      {label.isError && (
        <div className="mx-3 rounded border border-danger/40 bg-danger/5 px-3 py-1.5 text-2xs text-danger">
          label failed: {label.error instanceof Error ? label.error.message : String(label.error)}
        </div>
      )}

      <div className="flex max-h-[calc(100vh-320px)] flex-col gap-2 overflow-auto px-3 pb-3">
        {isLoading ? (
          <Loading label="Loading candidates…" />
        ) : error ? (
          <ErrorState error={error} onRetry={() => void refetch()} />
        ) : candidates.length === 0 ? (
          <EmptyState
            title="No candidates to label"
            hint="The bootstrap proposer returned an empty queue. Run the eval harness to surface new candidates."
          />
        ) : (
          candidates.map((candidate, i) => (
            <CandidateCard
              key={candidate.candidate_id}
              ref={(el) => {
                cardRefs.current[i] = el;
              }}
              candidate={candidate}
              focused={i === cursor}
              saving={savingId === candidate.candidate_id}
              onChoose={(choice) => applyChoice(i, choice, true)}
              onFocus={() => setCursor(i)}
            />
          ))
        )}
      </div>
    </Panel>
  );
}

function ProgressBar({ reviewed, total }: { reviewed: number; total: number }) {
  const pct = total === 0 ? 0 : Math.round((reviewed / total) * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-bg">
        <div
          className="h-full rounded-full bg-accent transition-all duration-fast"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="font-mono text-2xs text-text-faint tabular-nums">{pct}%</span>
    </div>
  );
}
