/**
 * Recall playground (PRD §4.5) — the precision-debugging tool. Enter a query + scope,
 * run `recall()`, and inspect exactly what it returns: ranked results with relevance
 * <ScoreBar>s + the **why-it-surfaced** note, alongside a live preview of the
 * ~500-token SessionStart **injection index** that would be emitted (FR-RET-3).
 *
 * The scope field defaults from the top-bar scope (`useScope`). Submit is button- /
 * Enter-driven via the page-local `useRecallForm` (over the shared `useRecallMutation`).
 * Honors the dark observability theme: color-by-type result cards, struck/greyed
 * superseded hits, keyboard affordances (Enter to run, j/k to walk results, o to open).
 *
 * FE-B owns this file + everything under src/pages/recall/. It does not edit the
 * router, the api client/hooks, the theme, or the shared components.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type MutableRefObject } from "react";
import { useNavigate } from "react-router-dom";

import { Page } from "@/components/AppShell";
import {
  Button,
  EmptyState,
  ErrorState,
  Field,
  Input,
  KeyboardHints,
  Loading,
  Panel,
  Select,
  type KeyHint,
} from "@/components/index";
import type { ScoredMemory } from "@/api";
import { PATHS } from "@/routes";
import { pluralize } from "@/lib/format";

import {
  IndexPreview,
  ResultCard,
  useRecallForm,
  type RecallFormState,
} from "@/pages/recall/index";

const TOP_K_CHOICES = [5, 10, 20, 50];

const KEY_HINTS: KeyHint[] = [
  { keys: ["Enter"], label: "run recall" },
  { keys: ["j"], label: "next result" },
  { keys: ["k"], label: "prev result" },
  { keys: ["o"], label: "open selected" },
];

export default function Recall() {
  const form = useRecallForm();
  const navigate = useNavigate();
  const results = useMemo<ScoredMemory[]>(() => form.data?.results ?? [], [form.data]);

  // Normalize raw recall scores into [0,1] for the ScoreBar (scores may exceed 1).
  const maxScore = useMemo(
    () => results.reduce((m, r) => Math.max(m, r.score), 0),
    [results],
  );
  const norm = useCallback(
    (s: number) => (maxScore > 1 ? s / maxScore : Math.max(0, Math.min(1, s))),
    [maxScore],
  );

  // j/k keyboard selection over the results list.
  const [selected, setSelected] = useState(0);
  const cardRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    setSelected(0);
  }, [form.data]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      const typing =
        tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || target?.isContentEditable;
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
      if (results.length === 0) return;
      if (e.key === "j") {
        e.preventDefault();
        setSelected((i) => {
          const next = Math.min(results.length - 1, i + 1);
          cardRefs.current[next]?.scrollIntoView({ block: "nearest" });
          return next;
        });
      } else if (e.key === "k") {
        e.preventDefault();
        setSelected((i) => {
          const next = Math.max(0, i - 1);
          cardRefs.current[next]?.scrollIntoView({ block: "nearest" });
          return next;
        });
      } else if (e.key === "o") {
        e.preventDefault();
        const hit = results[selected];
        if (hit) navigate(PATHS.memoryDetail(hit.memory.id));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [results, selected, navigate]);

  return (
    <Page
      title="Recall Playground"
      subtitle="run recall() and preview the SessionStart injection index"
    >
      <div className="mx-auto flex max-w-6xl flex-col gap-4">
        <QueryBar form={form} />

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_minmax(0,420px)]">
          {/* Ranked results -------------------------------------------------- */}
          <Panel
            title={
              form.data ? `Results · ${pluralize(results.length, "hit", "hits")}` : "Results"
            }
            bodyClassName="flex flex-col gap-2"
          >
            <ResultsBody
              form={form}
              results={results}
              selected={selected}
              norm={norm}
              cardRefs={cardRefs}
            />
          </Panel>

          {/* Injection index preview ----------------------------------------- */}
          <Panel title="SessionStart index preview" className="self-start">
            <IndexPreviewBody form={form} />
          </Panel>
        </div>

        <KeyboardHints hints={KEY_HINTS} className="mx-auto pt-1" />
      </div>
    </Page>
  );
}

function QueryBar({ form }: { form: RecallFormState }) {
  return (
    <Panel title="Query">
      <form
        className="flex flex-col gap-3"
        onSubmit={(e) => {
          e.preventDefault();
          form.run();
        }}
      >
        <div className="grid grid-cols-1 gap-3 md:grid-cols-[1fr_220px_120px]">
          <Field label="Query">
            <Input
              value={form.query}
              autoFocus
              spellCheck={false}
              placeholder="what would the agent ask for…"
              onChange={(e) => form.setQuery(e.target.value)}
              className="w-full"
            />
          </Field>
          <Field label="Scope">
            <Input
              value={form.scope}
              spellCheck={false}
              placeholder="all scopes"
              onChange={(e) => form.setScope(e.target.value)}
              className="w-full font-mono"
            />
          </Field>
          <Field label="Top K">
            <Select
              value={form.topK}
              onChange={(e) => form.setTopK(Number(e.target.value))}
              className="w-full"
            >
              {TOP_K_CHOICES.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </Select>
          </Field>
        </div>

        <div className="flex items-center justify-between gap-3">
          <label className="flex cursor-pointer items-center gap-2 text-xs text-text-muted">
            <input
              type="checkbox"
              checked={form.includeIndexPreview}
              onChange={(e) => form.setIncludeIndexPreview(e.target.checked)}
              className="h-3.5 w-3.5 accent-accent"
            />
            include index preview
          </label>
          <div className="flex items-center gap-2">
            {form.data && (
              <Button type="button" variant="ghost" onClick={form.reset} disabled={form.pending}>
                clear
              </Button>
            )}
            <Button
              type="submit"
              variant="primary"
              loading={form.pending}
              disabled={!form.query.trim()}
            >
              Recall
            </Button>
          </div>
        </div>
      </form>
    </Panel>
  );
}

function ResultsBody({
  form,
  results,
  selected,
  norm,
  cardRefs,
}: {
  form: RecallFormState;
  results: ScoredMemory[];
  selected: number;
  norm: (s: number) => number;
  cardRefs: MutableRefObject<(HTMLDivElement | null)[]>;
}) {
  if (form.pending) return <Loading label="Running recall…" />;
  if (form.error) return <ErrorState error={form.error} onRetry={form.run} />;
  if (!form.data) {
    return (
      <EmptyState
        title="No recall yet"
        hint="Enter a query and run recall() to see ranked results with scores + why-it-surfaced notes."
      />
    );
  }
  if (results.length === 0) {
    return (
      <EmptyState
        title="No hits"
        hint={`recall("${form.lastReq?.query ?? ""}") returned nothing for this scope.`}
      />
    );
  }
  return (
    <>
      {results.map((scored, i) => (
        <ResultCard
          key={scored.memory.id}
          ref={(el) => {
            cardRefs.current[i] = el;
          }}
          rank={i + 1}
          scored={scored}
          selected={i === selected}
          scoreNorm={norm(scored.score)}
        />
      ))}
    </>
  );
}

function IndexPreviewBody({ form }: { form: RecallFormState }) {
  if (!form.includeIndexPreview) {
    return <EmptyState title="Preview disabled" hint="Enable “include index preview” to see the injection." />;
  }
  if (form.pending) return <Loading label="Building index…" />;
  if (!form.data) {
    return (
      <EmptyState
        title="Run a recall"
        hint="The ~500-token SessionStart index that would be injected appears here."
      />
    );
  }
  if (!form.data.index_preview) {
    return <EmptyState title="No index preview" hint="The backend returned no injection index for this query." />;
  }
  return <IndexPreview preview={form.data.index_preview} />;
}
