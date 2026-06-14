/**
 * Eval (PRD §4.8) — eval results + the F4 browser bootstrap-labeling workflow.
 *
 * Two stacked sections so the operator can read the harness results and label the gold
 * set on one screen:
 *   1. Results — precision / classifier accuracy / latency / no-leak / scaling metric
 *                cards with pass/fail vs threshold.  [useEval]
 *   2. Bootstrap labeling (F4) — keyboard-first candidate cards labeled preference /
 *                project_fact / idea_seed / not-a-memory, persisted via the mutation
 *                hooks, then folded into the gold set on finish.  [useBootstrapCandidates,
 *                useLabelBootstrap, useFinishBootstrap]
 *
 * This file owns only its page + the page-local components under pages/eval/**. It
 * consumes shared design-system components and the typed api hooks; it edits no
 * shared/contract files.
 */

import { Page } from "@/components";
import { EvalMetrics } from "@/pages/eval/EvalMetrics";
import { BootstrapLabeler } from "@/pages/eval/BootstrapLabeler";

export default function Eval() {
  return (
    <Page
      title="Eval"
      subtitle="Harness results · F4 bootstrap labeling"
      bodyClassName="flex flex-col gap-4"
    >
      <EvalMetrics />
      <BootstrapLabeler />
    </Page>
  );
}
