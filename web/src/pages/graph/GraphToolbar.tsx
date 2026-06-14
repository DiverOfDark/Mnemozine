/**
 * Graph explorer — control toolbar (PRD §4.4).
 *
 * Entity-type filter, traversal depth, layout selector, and the cross-reference
 * overlay toggle. Controlled by useGraphFilters; the page owns the state. The
 * crossref toggle is styled rose (the crossref overlay color) when on, so the
 * control matches the highlight it produces.
 */

import type { GraphLayoutName } from "@/components";
import { Button, Field, Select } from "@/components";
import { HEX } from "@/theme/tokens";
import { cn } from "@/lib/cn";
import { ENTITY_TYPE_OPTIONS, LAYOUT_OPTIONS } from "./useGraphFilters";

interface GraphToolbarProps {
  entityType: string;
  depth: number;
  layout: GraphLayoutName;
  includeCrossrefs: boolean;
  focusEntity: string | null;
  onEntityType: (value: string) => void;
  onDepth: (value: number) => void;
  onLayout: (value: GraphLayoutName) => void;
  onToggleCrossrefs: (value: boolean) => void;
  onClearFocus: () => void;
}

const DEPTH_OPTIONS = [1, 2, 3];

export function GraphToolbar({
  entityType,
  depth,
  layout,
  includeCrossrefs,
  focusEntity,
  onEntityType,
  onDepth,
  onLayout,
  onToggleCrossrefs,
  onClearFocus,
}: GraphToolbarProps) {
  return (
    <div className="flex flex-wrap items-end gap-3 border-b border-border bg-bg-raised px-5 py-3">
      <Field label="entity type" className="w-40">
        <Select value={entityType} onChange={(e) => onEntityType(e.target.value)}>
          {ENTITY_TYPE_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </Select>
      </Field>

      <Field label="depth" className="w-24">
        <Select value={String(depth)} onChange={(e) => onDepth(Number(e.target.value))}>
          {DEPTH_OPTIONS.map((d) => (
            <option key={d} value={d}>
              {d} hop{d > 1 ? "s" : ""}
            </option>
          ))}
        </Select>
      </Field>

      <Field label="layout" className="w-44">
        <Select value={layout} onChange={(e) => onLayout(e.target.value as GraphLayoutName)}>
          {LAYOUT_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </Select>
      </Field>

      <button
        type="button"
        onClick={() => onToggleCrossrefs(!includeCrossrefs)}
        aria-pressed={includeCrossrefs}
        title="Toggle cross-reference overlay (rose dashed edges)"
        className={cn(
          "mb-0.5 inline-flex items-center gap-2 rounded border px-2.5 py-1 text-xs font-medium transition-colors duration-fast",
          includeCrossrefs
            ? "bg-bg-inset"
            : "border-border-strong bg-bg-inset text-text-muted hover:bg-bg-hover hover:text-text",
        )}
        style={
          includeCrossrefs
            ? { color: HEX.crossref, borderColor: HEX.crossref }
            : undefined
        }
      >
        <span
          className="h-1.5 w-3 shrink-0 rounded-full"
          style={{
            backgroundColor: includeCrossrefs ? HEX.crossref : HEX.textFaint,
            opacity: includeCrossrefs ? 1 : 0.6,
          }}
        />
        cross-refs
      </button>

      {focusEntity && (
        <Button variant="ghost" onClick={onClearFocus} className="mb-0.5" title={`focused on ${focusEntity}`}>
          clear focus
        </Button>
      )}
    </div>
  );
}
