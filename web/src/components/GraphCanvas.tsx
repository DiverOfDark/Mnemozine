/**
 * GraphCanvas — a thin Cytoscape.js wrapper for the Graph explorer (PRD §4.4).
 *
 * Takes the API's GraphResponse ({nodes, edges}) and renders a dark-themed,
 * interactive graph. Node color follows kind/entity (entity vs idea_seed vs
 * memory); cross-reference edges (is_crossref) are highlighted in rose with a
 * dashed line and carry their human-readable reason as a tooltip/title (FR-RET-6).
 * Superseded (inactive) edges render faded.
 *
 * The Graph screen agent supplies the data + onNodeSelect/onEdgeSelect callbacks
 * and chooses the layout; this component owns the Cytoscape lifecycle + stylesheet
 * so the look is consistent and the agent never touches raw Cytoscape styling.
 */

import { useEffect, useMemo, useRef } from "react";
import cytoscape, {
  type Core,
  type Css,
  type EdgeSingular,
  type ElementDefinition,
  type LayoutOptions,
  type NodeSingular,
  type Stylesheet,
} from "cytoscape";
import type { GraphResponse } from "@/api/types";
import { HEX } from "@/theme/tokens";
import { cn } from "@/lib/cn";

export type GraphLayoutName = "cose" | "concentric" | "grid" | "circle" | "breadthfirst";

interface GraphCanvasProps {
  data: GraphResponse | undefined;
  onNodeSelect?: (nodeId: string) => void;
  onEdgeSelect?: (edgeId: string) => void;
  /** Layout algorithm (default "cose"). */
  layout?: GraphLayoutName;
  /** Highlight a node id (e.g. the focused entity). */
  highlightNodeId?: string | null;
  className?: string;
}

/** Map a GraphResponse to Cytoscape element definitions. */
function toElements(data: GraphResponse): ElementDefinition[] {
  const nodes: ElementDefinition[] = data.nodes.map((n) => ({
    group: "nodes",
    data: {
      id: n.id,
      label: n.label,
      kind: n.kind,
      entityType: n.entity_type ?? "",
      scope: n.scope ?? "",
      memoryCount: n.memory_count,
    },
  }));
  const edges: ElementDefinition[] = data.edges.map((e) => ({
    group: "edges",
    data: {
      id: e.id,
      source: e.source,
      target: e.target,
      relation: e.relation,
      weight: e.weight,
      active: e.active,
      isCrossref: e.is_crossref,
      reason: e.reason ?? "",
    },
  }));
  return [...nodes, ...edges];
}

/** The dark-console Cytoscape stylesheet (token-driven). */
function stylesheet(): Stylesheet[] {
  const nodeStyle: Css.Node = {
    "background-color": (ele: NodeSingular) => {
      const kind = ele.data("kind");
      if (kind === "idea_seed") return HEX.type.idea_seed;
      if (kind === "memory") return HEX.type.project_fact;
      return HEX.accent; // entity
    },
    label: "data(label)",
    color: HEX.text,
    "font-size": 9,
    "font-family": "JetBrains Mono, monospace",
    "text-valign": "bottom",
    "text-margin-y": 4,
    "text-max-width": "120px",
    "text-wrap": "ellipsis",
    width: (ele: NodeSingular) => 14 + Math.min(20, Number(ele.data("memoryCount")) * 2),
    height: (ele: NodeSingular) => 14 + Math.min(20, Number(ele.data("memoryCount")) * 2),
    "border-width": 1,
    "border-color": HEX.bgRaised,
  };

  const nodeSelected: Css.Node = {
    "border-width": 2,
    "border-color": HEX.text,
    "background-color": HEX.accent,
  };

  const edgeStyle: Css.Edge = {
    width: (ele: EdgeSingular) => 1 + Math.min(4, Number(ele.data("weight")) * 2),
    "line-color": HEX.edge,
    "target-arrow-color": HEX.edge,
    "target-arrow-shape": "triangle",
    "arrow-scale": 0.7,
    "curve-style": "bezier",
    opacity: (ele: EdgeSingular) => (ele.data("active") ? 0.85 : 0.3),
    label: "data(relation)",
    color: HEX.textFaint,
    "font-size": 7,
    "font-family": "JetBrains Mono, monospace",
    "text-rotation": "autorotate",
  };

  // Cross-reference overlay edges (FR-RET-6): rose, dashed, reason carried in data.
  const crossrefEdge: Css.Edge = {
    "line-color": HEX.crossref,
    "target-arrow-color": HEX.crossref,
    "line-style": "dashed",
    width: 2,
    opacity: 0.9,
  };

  const edgeSelected: Css.Edge = {
    "line-color": HEX.text,
    "target-arrow-color": HEX.text,
    opacity: 1,
  };

  return [
    { selector: "node", style: nodeStyle },
    { selector: "node:selected, node.highlighted", style: nodeSelected },
    { selector: "edge", style: edgeStyle },
    { selector: "edge[?isCrossref]", style: crossrefEdge },
    { selector: "edge:selected", style: edgeSelected },
  ];
}

function layoutOptions(name: GraphLayoutName): LayoutOptions {
  switch (name) {
    case "cose":
      return {
        name: "cose",
        fit: true,
        padding: 30,
        animate: false,
        idealEdgeLength: () => 90,
        nodeRepulsion: () => 6000,
      };
    case "concentric":
      return {
        name: "concentric",
        fit: true,
        padding: 30,
        concentric: (n: NodeSingular) => Number(n.data("memoryCount")) || 1,
        levelWidth: () => 2,
      };
    case "grid":
      return { name: "grid", fit: true, padding: 30 };
    case "circle":
      return { name: "circle", fit: true, padding: 30 };
    case "breadthfirst":
      return { name: "breadthfirst", fit: true, padding: 30 };
  }
}

export function GraphCanvas({
  data,
  onNodeSelect,
  onEdgeSelect,
  layout = "cose",
  highlightNodeId,
  className,
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const elements = useMemo(() => (data ? toElements(data) : []), [data]);

  // Init Cytoscape once.
  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      style: stylesheet(),
      minZoom: 0.2,
      maxZoom: 3,
      wheelSensitivity: 0.2,
    });
    cyRef.current = cy;
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  // Wire selection callbacks (kept in a separate effect so they can update).
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    const onNode = (e: cytoscape.EventObject) => onNodeSelect?.(e.target.id());
    const onEdge = (e: cytoscape.EventObject) => onEdgeSelect?.(e.target.id());
    cy.on("tap", "node", onNode);
    cy.on("tap", "edge", onEdge);
    return () => {
      cy.off("tap", "node", onNode);
      cy.off("tap", "edge", onEdge);
    };
  }, [onNodeSelect, onEdgeSelect]);

  // Replace elements + relayout when data changes.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.batch(() => {
      cy.elements().remove();
      cy.add(elements);
    });
    // Add reason as a native title for crossref edges (hover tooltip).
    cy.edges("[?isCrossref]").forEach((edge) => {
      const reason = edge.data("reason");
      if (reason) edge.scratch("_reason", reason);
    });
    if (elements.length > 0) {
      cy.layout(layoutOptions(layout)).run();
    }
  }, [elements, layout]);

  // Apply highlight.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.nodes().removeClass("highlighted");
    if (highlightNodeId) {
      cy.getElementById(highlightNodeId).addClass("highlighted");
    }
  }, [highlightNodeId]);

  return (
    <div
      ref={containerRef}
      className={cn("h-full w-full rounded-md border border-border bg-bg", className)}
    />
  );
}
