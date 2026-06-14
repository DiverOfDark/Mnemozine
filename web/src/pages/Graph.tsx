/**
 * Graph explorer (PRD §4.4, FE-C) — an interactive Cytoscape canvas of entities,
 * idea-seeds and memories with weighted relation edges. Click a node to inspect its
 * memories + neighborhood; cross-reference connections are highlighted (rose, dashed)
 * with their mandatory human-readable reason (FR-RET-6). Filter by scope (top bar),
 * entity-type, depth, and toggle the crossref overlay.
 *
 * Wires useGraph (scope from useScope); the node drawer fans out to useMemories +
 * useCrossRefs. This file owns only itself + src/pages/graph/**; it never touches the
 * router, api client, theme, or shared components.
 */

import { useCallback, useMemo, useState } from "react";
import type { GraphEdge, GraphNode } from "@/api";
import { useGraph } from "@/api";
import {
  Badge,
  ErrorState,
  GraphCanvas,
  KeyboardHints,
  Loading,
  PageHeader,
} from "@/components";
import { useScope } from "@/state/scope";
import { HEX } from "@/theme/tokens";
import { GraphLegend } from "./graph/GraphLegend";
import { GraphToolbar } from "./graph/GraphToolbar";
import { NodeInspector } from "./graph/NodeInspector";
import { useGraphFilters } from "./graph/useGraphFilters";

export default function Graph() {
  const { scope } = useScope();
  const {
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
  } = useGraphFilters(scope);

  const graph = useGraph(query, { placeholderData: (prev) => prev });

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);

  const nodeById = useMemo(() => {
    const map = new Map<string, GraphNode>();
    for (const n of graph.data?.nodes ?? []) map.set(n.id, n);
    return map;
  }, [graph.data]);

  const edgeById = useMemo(() => {
    const map = new Map<string, GraphEdge>();
    for (const e of graph.data?.edges ?? []) map.set(e.id, e);
    return map;
  }, [graph.data]);

  const selectedNode = selectedNodeId ? nodeById.get(selectedNodeId) ?? null : null;
  const selectedEdge = selectedEdgeId ? edgeById.get(selectedEdgeId) ?? null : null;

  const handleNodeSelect = useCallback((id: string) => {
    setSelectedNodeId(id);
    setSelectedEdgeId(null);
  }, []);

  const handleEdgeSelect = useCallback((id: string) => {
    setSelectedEdgeId(id);
  }, []);

  const handleFocus = useCallback(
    (entity: string) => {
      setFocusEntity(entity);
      setSelectedNodeId(null);
    },
    [setFocusEntity],
  );

  const nodeCount = graph.data?.nodes.length ?? 0;
  const edgeCount = graph.data?.edges.length ?? 0;
  const crossrefCount = useMemo(
    () => (graph.data?.edges ?? []).filter((e) => e.is_crossref).length,
    [graph.data],
  );
  const truncated = graph.data?.truncated ?? false;

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        title="Graph Explorer"
        subtitle={
          focusEntity
            ? `focused on ${focusEntity} · ${nodeCount} nodes · ${edgeCount} edges`
            : `${nodeCount} nodes · ${edgeCount} edges · ${crossrefCount} cross-refs`
        }
        actions={
          <KeyboardHints
            hints={[
              { keys: ["click"], label: "inspect node" },
              { keys: ["Esc"], label: "close panel" },
            ]}
          />
        }
      />

      <GraphToolbar
        entityType={entityType}
        depth={depth}
        layout={layout}
        includeCrossrefs={includeCrossrefs}
        focusEntity={focusEntity}
        onEntityType={setEntityType}
        onDepth={setDepth}
        onLayout={setLayout}
        onToggleCrossrefs={setIncludeCrossrefs}
        onClearFocus={() => setFocusEntity(null)}
      />

      <div className="relative min-h-0 flex-1 p-3">
        {graph.isLoading && !graph.data ? (
          <Loading label="loading graph…" />
        ) : graph.error ? (
          <ErrorState error={graph.error} onRetry={() => void graph.refetch()} />
        ) : nodeCount === 0 ? (
          <div className="flex h-full items-center justify-center">
            <div className="max-w-md text-center">
              <div className="text-sm text-text-muted">No graph for this scope</div>
              <div className="mt-1 text-xs text-text-faint">
                Try widening the scope in the top bar, raising the depth, or clearing the entity-type
                filter.
              </div>
            </div>
          </div>
        ) : (
          <>
            <GraphCanvas
              data={graph.data}
              layout={layout}
              onNodeSelect={handleNodeSelect}
              onEdgeSelect={handleEdgeSelect}
              highlightNodeId={selectedNodeId}
            />
            <GraphLegend />

            {truncated && (
              <div className="absolute right-3 top-3 z-10 rounded-md border border-border bg-bg-raised/90 px-2.5 py-1 backdrop-blur-sm">
                <span className="text-2xs text-warn">graph truncated — narrow the scope to see all</span>
              </div>
            )}

            {selectedEdge && (
              <EdgeReasonPanel edge={selectedEdge} onClose={() => setSelectedEdgeId(null)} />
            )}
          </>
        )}
      </div>

      <NodeInspector
        node={selectedNode}
        scope={scope}
        isFocused={Boolean(selectedNode && focusEntity === selectedNode.label)}
        onClose={() => setSelectedNodeId(null)}
        onFocus={handleFocus}
      />
    </div>
  );
}

/**
 * A small floating panel that surfaces a selected edge's relation/weight and — for a
 * cross-reference edge — its human-readable reason (FR-RET-6) in the rose accent.
 */
function EdgeReasonPanel({ edge, onClose }: { edge: GraphEdge; onClose: () => void }) {
  return (
    <div
      className="absolute bottom-3 right-3 z-10 max-w-sm rounded-md border bg-bg-raised/95 px-3 py-2.5 backdrop-blur-sm"
      style={{ borderColor: edge.is_crossref ? HEX.crossref : HEX.borderStrong }}
    >
      <div className="mb-1 flex items-center gap-2">
        {edge.is_crossref ? (
          <span
            className="font-mono text-2xs uppercase tracking-wide"
            style={{ color: HEX.crossref }}
          >
            cross-reference
          </span>
        ) : (
          <span className="font-mono text-2xs uppercase tracking-wide text-text-muted">
            {edge.relation || "relation"}
          </span>
        )}
        {!edge.active && (
          <Badge bgClass="bg-tier-bg-archive" textClass="text-superseded">
            inactive
          </Badge>
        )}
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className="ml-auto text-text-faint hover:text-text"
        >
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
          </svg>
        </button>
      </div>
      {edge.reason ? (
        <p className="text-xs italic text-text">“{edge.reason}”</p>
      ) : (
        <p className="text-xs text-text-muted">
          weight {edge.weight.toFixed(2)}
          {edge.relation && ` · ${edge.relation}`}
        </p>
      )}
      {edge.reason && (
        <p className="mt-1 font-mono text-2xs text-text-faint">weight {edge.weight.toFixed(2)}</p>
      )}
    </div>
  );
}
