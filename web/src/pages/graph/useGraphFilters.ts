/**
 * Graph explorer — local filter/control state (FE-C / PRD §4.4).
 *
 * Owns the GraphQuery that drives useGraph (scope from the top bar, entity-type
 * filter, traversal depth, crossref overlay toggle) plus the focused entity and the
 * chosen Cytoscape layout. Kept page-local because no other screen needs the graph
 * control shape.
 */

import { useMemo, useState } from "react";
import type { GraphQuery } from "@/api";
import type { GraphLayoutName } from "@/components";

/** Entity-type filter options. "" = all. The backend treats unknown as "all". */
export const ENTITY_TYPE_OPTIONS = [
  { value: "", label: "all types" },
  { value: "entity", label: "entities" },
  { value: "idea_seed", label: "idea seeds" },
  { value: "memory", label: "memories" },
] as const;

export const LAYOUT_OPTIONS: { value: GraphLayoutName; label: string }[] = [
  { value: "cose", label: "force (cose)" },
  { value: "concentric", label: "concentric" },
  { value: "breadthfirst", label: "breadth-first" },
  { value: "circle", label: "circle" },
  { value: "grid", label: "grid" },
];

export const GRAPH_DEFAULT_DEPTH = 1;
export const GRAPH_DEFAULT_LIMIT = 150;

export interface UseGraphFiltersResult {
  entityType: string;
  depth: number;
  includeCrossrefs: boolean;
  layout: GraphLayoutName;
  /** The focused entity id used to re-scope the subgraph (graph `entity` param). */
  focusEntity: string | null;
  query: GraphQuery;
  setEntityType: (value: string) => void;
  setDepth: (value: number) => void;
  setIncludeCrossrefs: (value: boolean) => void;
  setLayout: (value: GraphLayoutName) => void;
  setFocusEntity: (value: string | null) => void;
}

export function useGraphFilters(scope: string | null): UseGraphFiltersResult {
  const [entityType, setEntityType] = useState("");
  const [depth, setDepth] = useState(GRAPH_DEFAULT_DEPTH);
  const [includeCrossrefs, setIncludeCrossrefs] = useState(true);
  const [layout, setLayout] = useState<GraphLayoutName>("cose");
  const [focusEntity, setFocusEntity] = useState<string | null>(null);

  const query = useMemo<GraphQuery>(
    () => ({
      scope: scope ?? undefined,
      entity: focusEntity ?? undefined,
      entity_type: entityType || undefined,
      depth,
      include_crossrefs: includeCrossrefs,
      limit: GRAPH_DEFAULT_LIMIT,
    }),
    [scope, focusEntity, entityType, depth, includeCrossrefs],
  );

  return {
    entityType,
    depth,
    includeCrossrefs,
    layout,
    focusEntity,
    query,
    setEntityType,
    setDepth,
    setIncludeCrossrefs,
    setLayout,
    setFocusEntity,
  };
}
