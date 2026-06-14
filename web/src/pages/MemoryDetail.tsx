/**
 * Memory detail (PRD §4.3) — the full single-memory view and the home of the
 * temporal/supersession signature feature (PRD §2).
 *
 * Shows: full content (mono); classification with a **reclassify** control; scope
 * with a **re-scope** control; entity chips → graph; confidence + access stats;
 * the **provenance** link to the source session/raw transcript; the
 * **validity-window timeline**; the **supersession chain** (replaced / replaced-by);
 * and the tier with **archive/restore**. All HITL writes (R1/R5) go through the
 * shared, cache-invalidating `usePatchMemory` mutation via the page-local
 * `usePatchControls` hook.
 *
 * Routed full-page at `/memories/:memoryId`. Honors the dark observability theme:
 * color-by-type accent, struck/greyed superseded state, keyboard affordances.
 *
 * FE-B owns this file + everything under src/pages/memory-detail/. It does not edit
 * the router, the api client/hooks, the theme, or the shared components.
 */

import { useCallback, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { Page } from "@/components/AppShell";
import {
  Button,
  ErrorState,
  KeyboardHints,
  Loading,
  Panel,
  StatusBadge,
  SupersessionChain,
  TierBadge,
  TypeBadge,
  ValidityTimeline,
  type KeyHint,
} from "@/components/index";
import { useMemory } from "@/api";
import type { MemoryDetail as MemoryDetailType } from "@/api";
import { PATHS } from "@/routes";
import { cn } from "@/lib/cn";
import { HEX } from "@/theme/tokens";

import {
  AccessStats,
  CorrectionControls,
  EntityChips,
  ProvenanceBlock,
  usePatchControls,
} from "@/pages/memory-detail/index";

const KEY_HINTS: KeyHint[] = [
  { keys: ["e"], label: "archive / restore" },
  { keys: ["g"], label: "back to memories" },
  { keys: ["Esc"], label: "back" },
];

export default function MemoryDetail() {
  const { memoryId } = useParams<{ memoryId: string }>();
  const navigate = useNavigate();
  const query = useMemory(memoryId);
  const memory = query.data;
  const controls = usePatchControls(memory);

  const goBack = useCallback(() => navigate(PATHS.memories), [navigate]);
  const goToMemory = useCallback(
    (id: string) => navigate(PATHS.memoryDetail(id)),
    [navigate],
  );

  // Keyboard affordances (skip while typing into an input/select/textarea).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || target?.isContentEditable) {
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "Escape" || e.key === "g") {
        e.preventDefault();
        goBack();
      } else if (e.key === "e" && memory) {
        e.preventDefault();
        controls.toggleTier(memory.tier);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [goBack, controls, memory]);

  const headerActions = (
    <Button variant="ghost" onClick={goBack} title="Back to Memories (g / Esc)">
      ← Memories
    </Button>
  );

  if (query.isLoading) {
    return (
      <Page title="Memory" subtitle="loading…" actions={headerActions}>
        <Loading label="Loading memory…" />
      </Page>
    );
  }

  if (query.isError || !memory) {
    return (
      <Page title="Memory" subtitle={memoryId} actions={headerActions}>
        <ErrorState error={query.error ?? new Error("Memory not found")} onRetry={() => void query.refetch()} />
      </Page>
    );
  }

  return (
    <Page
      title={
        <span className="flex items-center gap-2.5">
          <span
            className="h-2.5 w-2.5 shrink-0 rounded-full"
            style={{ backgroundColor: HEX.type[memory.type] }}
            aria-hidden="true"
          />
          Memory
        </span>
      }
      subtitle={<span className="font-mono">{memory.id}</span>}
      actions={headerActions}
    >
      <Body memory={memory} controls={controls} onSelect={goToMemory} />
    </Page>
  );
}

function Body({
  memory,
  controls,
  onSelect,
}: {
  memory: MemoryDetailType;
  controls: ReturnType<typeof usePatchControls>;
  onSelect: (id: string) => void;
}) {
  const active = memory.validity.active;
  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-4">
      {/* Content + classification header ----------------------------------- */}
      <Panel
        title="Content"
        actions={
          <div className="flex items-center gap-1.5">
            <TypeBadge type={memory.type} />
            <TierBadge tier={memory.tier} />
            <StatusBadge active={active} />
          </div>
        }
      >
        <p
          className={cn(
            "whitespace-pre-wrap break-words font-mono text-sm leading-relaxed",
            active ? "text-text" : "superseded",
          )}
        >
          {memory.content}
        </p>
        <div className="mt-3 border-t border-border pt-3">
          <span className="mb-2 block text-2xs font-semibold uppercase tracking-wider text-text-faint">
            Entities
          </span>
          <EntityChips entities={memory.entities} />
        </div>
      </Panel>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Left column ----------------------------------------------------- */}
        <div className="flex flex-col gap-4">
          <Panel title="Corrections (HITL)">
            <CorrectionControls memory={memory} controls={controls} />
          </Panel>

          <Panel title="Validity window">
            <ValidityTimeline validity={memory.validity} />
          </Panel>

          <Panel title="Access stats">
            <AccessStats memory={memory} />
          </Panel>
        </div>

        {/* Right column ---------------------------------------------------- */}
        <div className="flex flex-col gap-4">
          <Panel title="Supersession chain">
            <SupersessionChain
              supersedes={memory.supersedes}
              supersededBy={memory.superseded_by}
              currentId={memory.id}
              currentContent={memory.content}
              onSelect={onSelect}
            />
          </Panel>

          <Panel title="Provenance">
            <ProvenanceBlock provenance={memory.provenance} />
          </Panel>
        </div>
      </div>

      <KeyboardHints hints={KEY_HINTS} className="mx-auto pt-1" />
    </div>
  );
}
