/**
 * CorrectionControls (FE-B / PRD §4.3, R1/R5 HITL) — the three operator corrections
 * on a memory:
 *   • reclassify — a type <Select> (preference / project_fact / idea_seed)
 *   • re-scope   — a scope <Input> committed on Enter / blur
 *   • archive/restore — a tier toggle button (hot ⇄ archive)
 *
 * All writes go through the page-local {@link usePatchControls} hook (which wraps the
 * shared cache-invalidating `usePatchMemory`). Content is intentionally NOT editable
 * here (PRD §7 — the patch contract accepts only type/scope/tier). Per-control
 * spinners + a `changed[]` echo / error line give honest write feedback.
 */

import { useEffect, useState } from "react";
import { Button, Field, Input, Select, Spinner } from "@/components/index";
import { MEMORY_TYPES } from "@/api";
import type { MemoryDetail } from "@/api";
import type { PatchControls } from "@/pages/memory-detail/usePatchControls";
import { TYPE_BADGE } from "@/theme/tokens";

export function CorrectionControls({
  memory,
  controls,
}: {
  memory: MemoryDetail;
  controls: PatchControls;
}) {
  const [scopeDraft, setScopeDraft] = useState(memory.scope);

  // Keep the scope draft in sync when the underlying memory changes (after a patch).
  useEffect(() => {
    setScopeDraft(memory.scope);
  }, [memory.scope]);

  const archived = memory.tier === "archive";
  const tierBusy = controls.pendingField === "tier";
  const typeBusy = controls.pendingField === "type";
  const scopeBusy = controls.pendingField === "scope";
  const scopeDirty = scopeDraft.trim() !== memory.scope && scopeDraft.trim().length > 0;

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {/* Reclassify --------------------------------------------------------- */}
        <Field label="Reclassify (type)">
          <div className="flex items-center gap-2">
            <Select
              value={memory.type}
              disabled={controls.pending}
              onChange={(e) => controls.reclassify(e.target.value as MemoryDetail["type"])}
              className="flex-1"
            >
              {MEMORY_TYPES.map((t) => (
                <option key={t} value={t}>
                  {TYPE_BADGE[t].label}
                </option>
              ))}
            </Select>
            {typeBusy && <Spinner size={12} />}
          </div>
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
              placeholder="global | project:<id>"
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
