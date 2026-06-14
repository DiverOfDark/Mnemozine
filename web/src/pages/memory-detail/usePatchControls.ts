/**
 * Page-local hook (FE-B / PRD §4.3) wrapping the three HITL corrections on a memory
 * — reclassify (`type`), re-scope (`scope`), archive/restore (`tier`) — over the
 * shared `usePatchMemory` mutation. It tracks which field is "in flight" so the
 * controls can show per-control spinners, and surfaces the last `changed[]` echo +
 * any error for an inline status line.
 *
 * The page owns these UI ergonomics; the actual write goes through the shared,
 * cache-invalidating mutation hook in @/api (we consume it, never re-implement it).
 */

import { useCallback, useState } from "react";
import { usePatchMemory } from "@/api";
import type { MemoryDetail, MemoryPatchRequest, MemoryType, Tier } from "@/api";

/** Which control triggered the in-flight write (for per-control spinners). */
export type PatchField = "type" | "scope" | "tier" | null;

export interface PatchControls {
  /** True while any patch is in flight. */
  pending: boolean;
  /** Which field is currently being written (drives per-control spinners). */
  pendingField: PatchField;
  /** Fields changed by the most recent successful patch (MutationResponse.changed). */
  lastChanged: string[];
  /** Error from the most recent patch, if it failed. */
  error: Error | null;
  reclassify: (type: MemoryType) => void;
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

  const reclassify = useCallback(
    (type: MemoryType) => {
      if (memory && type !== memory.type) run("type", { type });
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
    reclassify,
    rescope,
    setTier,
    toggleTier,
  };
}
