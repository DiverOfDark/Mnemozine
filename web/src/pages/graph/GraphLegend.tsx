/**
 * Graph explorer — a compact legend overlaid on the canvas (PRD §4.4 / §5).
 *
 * Decodes the GraphCanvas color scheme: entity (accent) · idea_seed (amber) · memory
 * (sky) nodes, weighted relation edges, and the rose dashed cross-reference overlay.
 * Colors are read from the HEX token map (the canvas can't use Tailwind classes, so
 * the legend mirrors it via the same sanctioned tokens) — never raw literals.
 */

import { HEX } from "@/theme/tokens";

interface SwatchProps {
  color: string;
  label: string;
  /** Render as a dashed line instead of a dot (for the crossref edge swatch). */
  edge?: boolean;
  dashed?: boolean;
}

function Swatch({ color, label, edge, dashed }: SwatchProps) {
  return (
    <span className="inline-flex items-center gap-1.5">
      {edge ? (
        <span
          className="inline-block h-0 w-4 shrink-0"
          style={{
            borderTopWidth: 2,
            borderTopStyle: dashed ? "dashed" : "solid",
            borderTopColor: color,
          }}
        />
      ) : (
        <span
          className="inline-block h-2 w-2 shrink-0 rounded-full"
          style={{ backgroundColor: color }}
        />
      )}
      <span className="text-2xs text-text-muted">{label}</span>
    </span>
  );
}

export function GraphLegend() {
  return (
    <div className="pointer-events-none absolute bottom-3 left-3 z-10 flex flex-col gap-1.5 rounded-md border border-border bg-bg-raised/90 px-3 py-2 backdrop-blur-sm">
      <div className="mb-0.5 text-2xs font-semibold uppercase tracking-wider text-text-faint">legend</div>
      <Swatch color={HEX.accent} label="entity" />
      <Swatch color={HEX.type.idea_seed} label="cross-ref seed" />
      <Swatch color={HEX.type.project_fact} label="memory" />
      <Swatch color={HEX.edge} label="relation (weighted)" edge />
      <Swatch color={HEX.crossref} label="cross-reference" edge dashed />
    </div>
  );
}
