/**
 * CorrectionControls (FE-B / PRD §4.3, R1/R5 HITL) — the operator corrections on a
 * memory:
 *   • re-label   — a FREE-FORM category <Input> committed on Enter / blur
 *   • cross-ref  — a cross-reference seed flag toggle (replaces the idea_seed type)
 *   • re-scope   — a hierarchical scope <Input> committed on Enter / blur
 *   • archive/restore — a tier toggle button (hot ⇄ archive)
 *
 * All writes go through the page-local {@link usePatchControls} hook (which wraps the
 * shared cache-invalidating `usePatchMemory`). Content is intentionally NOT editable
 * here (PRD §7 — the patch contract accepts only category / cross_ref_candidate /
 * scope / tier). The core data-model redesign replaced the fixed `type` <Select>
 * with the free-form category input + the cross-ref flag.
 */

import { useEffect, useState } from "react";
import { Button, Field, Input, Spinner } from "@/components/index";
import type { MemoryDetail } from "@/api";
import type { PatchControls } from "@/pages/memory-detail/usePatchControls";

export function CorrectionControls({
  memory,
  controls,
}: {
  memory: MemoryDetail;
  controls: PatchControls;
}) {
  const [scopeDraft, setScopeDraft] = useState(memory.scope);
  const [categoryDraft, setCategoryDraft] = useState(memory.category);

  // Keep the drafts in sync when the underlying memory changes (after a patch).
  useEffect(() => {
    setScopeDraft(memory.scope);
  }, [memory.scope]);
  useEffect(() => {
    setCategoryDraft(memory.category);
  }, [memory.category]);

  const archived = memory.tier === "archive";
  const tierBusy = controls.pendingField === "tier";
  const categoryBusy = controls.pendingField === "category";
  const crossRefBusy = controls.pendingField === "cross_ref_candidate";
  const scopeBusy = controls.pendingField === "scope";
  const scopeDirty = scopeDraft.trim() !== memory.scope && scopeDraft.trim().length > 0;
  const categoryDirty =
    categoryDraft.trim().toLowerCase() !== memory.category && categoryDraft.trim().length > 0;

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {/* Re-label (free-form category) ------------------------------------- */}
        <Field label="Re-label (category)">
          <form
            className="flex items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              controls.recategorize(categoryDraft);
            }}
          >
            <Input
              value={categoryDraft}
              disabled={controls.pending}
              spellCheck={false}
              placeholder="e.g. preference, decision, gotcha"
              onChange={(e) => setCategoryDraft(e.target.value)}
              onBlur={() => controls.recategorize(categoryDraft)}
              className="flex-1 font-mono"
            />
            {categoryBusy ? (
              <Spinner size={12} />
            ) : (
              <Button
                type="submit"
                variant="default"
                disabled={!categoryDirty || controls.pending}
              >
                set
              </Button>
            )}
          </form>
        </Field>

        {/* Re-scope ----------------------------------------------------------- */}
        <Field label="Re-scope">
          <form
            className="flex items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              controls.rescope(scopeDraft);
            }}
          >
            <Input
              value={scopeDraft}
              disabled={controls.pending}
              spellCheck={false}
              placeholder="global | project:<id>[/<sub>]"
              onChange={(e) => setScopeDraft(e.target.value)}
              onBlur={() => controls.rescope(scopeDraft)}
              className="flex-1 font-mono"
            />
            <Button type="submit" variant="default" disabled={!scopeDirty || controls.pending} loading={scopeBusy}>
              set
            </Button>
          </form>
        </Field>
      </div>

      {/* Cross-reference seed flag ------------------------------------------- */}
      <div className="flex items-center justify-between rounded border border-border bg-bg px-3 py-2">
        <div className="flex flex-col">
          <span className="text-2xs uppercase tracking-wide text-text-faint">
            Cross-reference seed
          </span>
          <span className="text-xs text-text-muted">
            Flag this memory as a serendipitous-connection seed (replaces idea_seed).
          </span>
        </div>
        <label className="flex items-center gap-2 font-mono text-xs text-text-muted">
          {crossRefBusy && <Spinner size={12} />}
          <input
            type="checkbox"
            checked={memory.cross_ref_candidate}
            disabled={controls.pending}
            onChange={(e) => controls.setCrossRef(e.target.checked)}
            className="h-3.5 w-3.5 accent-accent"
          />
          {memory.cross_ref_candidate ? "on" : "off"}
        </label>
      </div>

      {/* Archive / restore ---------------------------------------------------- */}
      <div className="flex items-center justify-between rounded border border-border bg-bg px-3 py-2">
        <div className="flex flex-col">
          <span className="text-2xs uppercase tracking-wide text-text-faint">Tier</span>
          <span className="text-xs text-text-muted">
            {archived
              ? "Cold — off the hot retrieval path. Restore to surface it again."
              : "Hot — on the live retrieval path. Archive to cool it down."}
          </span>
        </div>
        <Button
          variant={archived ? "primary" : "danger"}
          loading={tierBusy}
          disabled={controls.pending}
          onClick={() => controls.toggleTier(memory.tier)}
          title={archived ? "Restore to hot (e)" : "Archive (e)"}
        >
          {archived ? "Restore to hot" : "Archive"}
        </Button>
      </div>

      {/* Write feedback ------------------------------------------------------- */}
      {controls.error ? (
        <p className="font-mono text-2xs text-danger">patch failed — {controls.error.message}</p>
      ) : controls.lastChanged.length > 0 ? (
        <p className="font-mono text-2xs text-ok">
          updated · {controls.lastChanged.join(", ")}
        </p>
      ) : null}
    </div>
  );
}
