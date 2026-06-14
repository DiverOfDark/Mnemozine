/**
 * Graph explorer — the node-inspection drawer (PRD §4.4).
 *
 * When the operator clicks a node, this drawer shows the node's metadata, the
 * memories anchored to that entity (useMemories filtered by entity), and the entity's
 * cross-reference connections (useCrossRefs) — each with its mandatory human-readable
 * reason (FR-RET-6), rendered in the rose crossref accent. From here the operator can
 * re-focus the subgraph on the node or jump to any memory's detail.
 */

import { Link } from "react-router-dom";
import type { GraphNode } from "@/api";
import { useCrossRefs, useMemories } from "@/api";
import {
  Badge,
  Button,
  CategoryBadge,
  CrossRefBadge,
  DetailDrawer,
  DrawerSection,
  KeyValue,
  Loading,
  StatusBadge,
} from "@/components";
import { HEX } from "@/theme/tokens";
import { formatRelative, parseScope } from "@/lib/format";
import { PATHS } from "@/routes";
import { cn } from "@/lib/cn";

interface NodeInspectorProps {
  node: GraphNode | null;
  scope: string | null;
  isFocused: boolean;
  onClose: () => void;
  onFocus: (entity: string) => void;
}

const MEMORY_PAGE_LIMIT = 50;

export function NodeInspector({ node, scope, isFocused, onClose, onFocus }: NodeInspectorProps) {
  const entity = node?.label ?? "";
  const enabled = Boolean(node);

  const memories = useMemories(
    { entity, scope: scope ?? undefined, limit: MEMORY_PAGE_LIMIT },
    { enabled },
  );
  const crossrefs = useCrossRefs(
    {
      entity,
      project: parseScope(scope).project,
      limit: 25,
    },
    { enabled },
  );

  if (!node) return null;

  const items = memories.data?.items ?? [];
  const crossItems = crossrefs.data?.items ?? [];

  return (
    <DetailDrawer
      open={enabled}
      onClose={onClose}
      title={node.label}
      subtitle={node.id}
      headerActions={
        <Badge bgClass="bg-bg-inset" textClass="text-text-muted">
          {node.kind}
        </Badge>
      }
      footer={
        <div className="flex items-center justify-between">
          <span className="font-mono text-2xs text-text-faint">
            {node.memory_count} {node.memory_count === 1 ? "memory" : "memories"}
          </span>
          <Button
            variant={isFocused ? "default" : "primary"}
            onClick={() => onFocus(node.label)}
            disabled={isFocused}
            title="Re-scope the graph to this node's neighborhood"
          >
            {isFocused ? "focused" : "focus neighborhood"}
          </Button>
        </div>
      }
    >
      <DrawerSection title="node">
        <div className="rounded-md border border-border bg-bg-inset px-3 py-1">
          <KeyValue k="kind">
            <span className="font-mono">{node.kind}</span>
          </KeyValue>
          <KeyValue k="entity type">
            <span className="font-mono">{node.entity_type ?? "—"}</span>
          </KeyValue>
          <KeyValue k="scope">
            <span className="font-mono">{node.scope ?? "—"}</span>
          </KeyValue>
          <KeyValue k="memories">
            <span className="font-mono">{node.memory_count}</span>
          </KeyValue>
        </div>
      </DrawerSection>

      <DrawerSection title={`memories (${memories.data?.page.total ?? items.length})`}>
        {memories.isLoading ? (
          <Loading label="loading memories…" />
        ) : items.length === 0 ? (
          <p className="text-2xs text-text-faint">No memories anchored to this node.</p>
        ) : (
          <ul className="flex flex-col gap-1.5">
            {items.map((m) => (
              <li key={m.id}>
                <Link
                  to={PATHS.memoryDetail(m.id)}
                  className={cn(
                    "block rounded border border-border bg-bg-inset px-2.5 py-2 transition-colors hover:border-border-strong hover:bg-bg-hover",
                    !m.active && "superseded",
                  )}
                >
                  <div className="mb-1 flex items-center gap-2">
                    <CategoryBadge category={m.category} />
                    {m.cross_ref_candidate && <CrossRefBadge />}
                    <StatusBadge active={m.active} />
                    <span className="ml-auto font-mono text-2xs text-text-faint">
                      {formatRelative(m.last_accessed)}
                    </span>
                  </div>
                  <p className="line-clamp-3 text-xs text-text">{m.content}</p>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </DrawerSection>

      <DrawerSection title={`cross-references (${crossItems.length})`}>
        {crossrefs.isLoading ? (
          <Loading label="loading cross-refs…" />
        ) : crossItems.length === 0 ? (
          <p className="text-2xs text-text-faint">No cross-reference connections for this node.</p>
        ) : (
          <ul className="flex flex-col gap-2">
            {crossItems.map((cr) => (
              <li
                key={cr.memory.id}
                className={cn(
                  "rounded border bg-bg-inset px-2.5 py-2",
                  cr.suppressed && "opacity-50",
                )}
                style={{ borderColor: HEX.crossref }}
              >
                <div className="mb-1 flex items-center gap-2">
                  <span
                    className="font-mono text-2xs uppercase tracking-wide"
                    style={{ color: HEX.crossref }}
                  >
                    crossref
                  </span>
                  {cr.suppressed && (
                    <Badge bgClass="bg-tier-bg-archive" textClass="text-superseded">
                      suppressed
                    </Badge>
                  )}
                  <Link
                    to={PATHS.memoryDetail(cr.memory.id)}
                    className="ml-auto font-mono text-2xs text-accent hover:text-accent-hover hover:underline"
                  >
                    {cr.memory.id.slice(0, 8)}…
                  </Link>
                </div>
                <p className="mb-1 text-xs italic text-text-muted">“{cr.reason}”</p>
                {cr.shared_entities.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {cr.shared_entities.map((ent) => (
                      <Badge key={ent} bgClass="bg-bg" textClass="text-text-faint">
                        {ent}
                      </Badge>
                    ))}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </DrawerSection>
    </DetailDrawer>
  );
}
