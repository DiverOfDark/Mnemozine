/**
 * Page-local hook (FE-B / PRD §4.3) wrapping the HITL corrections on a memory —
 * re-label (`category`), toggle the cross-ref seed flag (`cross_ref_candidate`),
 * re-scope (`scope`), archive/restore (`tier`) — over the shared `usePatchMemory`
 * mutation. It tracks which field is "in flight" so the controls can show
 * per-control spinners, and surfaces the last `changed[]` echo + any error for an
 * inline status line.
 *
 * The old fixed-enum `type` reclassify is now a FREE-FORM `category` re-label plus
 * the `cross_ref_candidate` flag toggle (the core data-model redesign). The actual
 * write goes through the shared, cache-invalidating mutation hook in @/api.
 */

import { useCallback, useState } from "react";
import { usePatchMemory } from "@/api";
import type { MemoryDetail, MemoryPatchRequest, Tier } from "@/api";

/** Which control triggered the in-flight write (for per-control spinners). */
export type PatchField = "category" | "cross_ref_candidate" | "scope" | "tier" | null;

export interface PatchControls {
  /** True while any patch is in flight. */
  pending: boolean;
  /** Which field is currently being written (drives per-control spinners). */
  pendingField: PatchField;
  /** Fields changed by the most recent successful patch (MutationResponse.changed). */
  lastChanged: string[];
  /** Error from the most recent patch, if it failed. */
  error: Error | null;
  /** Re-label the free-form category. */
  recategorize: (category: string) => void;
  /** Toggle the cross-reference seed flag. */
  setCrossRef: (value: boolean) => void;
  rescope: (scope: string) => void;
  /** Archive (tier=archive) ⇄ restore (tier=hot). */
  setTier: (tier: Tier) => void;
  /** Toggle the current tier (archive ⇄ restore) — for the keyboard `e` affordance. */
  toggleTier: (current: Tier) => void;
}

export function usePatchControls(memory: MemoryDetail | undefined): PatchControls {
  const mutation = usePatchMemory();
  const [pendingField, setPendingField] = useState<PatchField>(null);
  const [lastChanged, setLastChanged] = useState<string[]>([]);

  const run = useCallback(
    (field: Exclude<PatchField, null>, patch: MemoryPatchRequest) => {
      if (!memory) return;
      setPendingField(field);
      setLastChanged([]);
      mutation.mutate(
        { id: memory.id, patch },
        {
          onSuccess: (data) => setLastChanged(data.changed),
          onSettled: () => setPendingField(null),
        },
      );
    },
    [memory, mutation],
  );

  const recategorize = useCallback(
    (category: string) => {
      const next = category.trim().toLowerCase();
      if (memory && next && next !== memory.category) run("category", { category: next });
    },
    [memory, run],
  );

  const setCrossRef = useCallback(
    (value: boolean) => {
      if (memory && value !== memory.cross_ref_candidate) {
        run("cross_ref_candidate", { cross_ref_candidate: value });
      }
    },
    [memory, run],
  );

  const rescope = useCallback(
    (scope: string) => {
      const next = scope.trim();
      if (memory && next && next !== memory.scope) run("scope", { scope: next });
    },
    [memory, run],
  );

  const setTier = useCallback(
    (tier: Tier) => {
      if (memory && tier !== memory.tier) run("tier", { tier });
    },
    [memory, run],
  );

  const toggleTier = useCallback(
    (current: Tier) => setTier(current === "hot" ? "archive" : "hot"),
    [setTier],
  );

  return {
    pending: mutation.isPending,
    pendingField,
    lastChanged,
    error: mutation.error,
    recategorize,
    setCrossRef,
    rescope,
    setTier,
    toggleTier,
  };
}
