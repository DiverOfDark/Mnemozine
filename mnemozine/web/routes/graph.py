"""Graph explorer route (PRD §4.4 / §6 GET /api/graph).

Builds a scoped subgraph for the Cytoscape view from the live store via the single
bounded :meth:`StorageBackend.graph_snapshot` read: entity nodes (optionally
centered on one entity's one-hop neighborhood, optionally type-filtered, capped at
``limit`` IN CYPHER) joined by their weighted, temporal relationship edges from a
SINGLE aggregate edge query (no per-entity N+1), plus ``idea_seed`` memory nodes
(first-class graph nodes that power cross-referencing, §7) linked to the entities
they mention — all embedding-free. Cross-reference connections (FR-RET-6) are a
:class:`CrossReferencer` concern overlaid on top as ``is_crossref`` edges carrying
their mandatory human-readable ``reason`` so the UI can highlight serendipitous
links distinctly from the structural snapshot.

Filters (PRD §4.4): ``entity`` centers the subgraph on one entity's neighborhood,
``entity_type`` filters entity nodes by category, ``scope`` bounds the idea-seed /
cross-reference overlay, ``include_crossrefs`` toggles the overlay, and ``limit``
caps the node count (``truncated`` signals a cap). Runs against the in-memory fake
in tests.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from mnemozine.interfaces import (
    CrossReferencer,
    GraphSnapshotEdge,
    GraphSnapshotNode,
    RetrievalContext,
)
from mnemozine.web.deps import CrossReferencerDep, StorageDep
from mnemozine.web.routes._read import scope_to_obj
from mnemozine.web.schemas import GraphEdge, GraphNode, GraphResponse

router = APIRouter(prefix="/api/graph", tags=["graph"])

# Stable node-id prefixes so entity and memory nodes never collide in Cytoscape.
_ENT_PREFIX = "ent:"
_MEM_PREFIX = "mem:"


def _node_id(node: GraphSnapshotNode) -> str:
    """Map a bare snapshot node id onto the prefixed Cytoscape node-id namespace."""

    prefix = _ENT_PREFIX if node.kind == "entity" else _MEM_PREFIX
    return f"{prefix}{node.id}"


def _snapshot_node(node: GraphSnapshotNode) -> GraphNode:
    """Project a :class:`GraphSnapshotNode` onto the Cytoscape wire node."""

    return GraphNode(
        id=_node_id(node),
        label=node.label,
        kind=node.kind,
        entity_type=node.entity_type,
        scope=node.scope,
        memory_count=node.memory_count,
    )


def _snapshot_edge(edge: GraphSnapshotEdge) -> GraphEdge:
    """Project a :class:`GraphSnapshotEdge` onto the Cytoscape wire edge.

    Structural ``relates`` edges connect two entities (``ent:`` prefix on both);
    ``mentions`` edges run from an idea-seed memory node (``mem:``) to an entity
    (``ent:``). Neither is a cross-reference — the FR-RET-6 overlay is added on top.
    """

    if edge.kind == "mentions":
        source = f"{_MEM_PREFIX}{edge.source}"
    else:
        source = f"{_ENT_PREFIX}{edge.source}"
    return GraphEdge(
        id=edge.id,
        source=source,
        target=f"{_ENT_PREFIX}{edge.target}",
        relation=edge.relation,
        weight=edge.weight,
        active=edge.active,
        is_crossref=False,
        reason=None,
    )


async def _crossref_overlay(
    cross_referencer: CrossReferencer,
    entity_names: list[str],
    scope_str: str | None,
    project: str | None,
) -> list[GraphEdge]:
    """Build the FR-RET-6 cross-reference overlay edges (with reasons).

    Runs ``find_related`` over a working context seeded from the subgraph's entity
    names and connects each surfaced memory back to its first shared entity, so the
    UI can draw an explained, highlighted serendipitous link. Best-effort: a
    cross-ref failure must not break the structural graph.
    """

    context = RetrievalContext(
        project=project,
        entities=entity_names,
    )
    try:
        refs = await cross_referencer.find_related(context)
    except Exception:  # noqa: BLE001 - the overlay is advisory; never break the graph
        return []

    overlay: list[GraphEdge] = []
    for i, ref in enumerate(refs):
        anchor = ref.shared_entities[0] if ref.shared_entities else None
        if anchor is None:
            continue
        overlay.append(
            GraphEdge(
                id=f"crossref:{ref.memory.id}:{i}",
                source=f"{_MEM_PREFIX}{ref.memory.id}",
                target=f"{_ENT_PREFIX}{anchor}",
                relation="cross_reference",
                weight=ref.score,
                active=True,
                is_crossref=True,
                reason=ref.reason,
            )
        )
    return overlay


@router.get("", response_model=GraphResponse, summary="Scoped subgraph for the explorer")
async def get_graph(
    storage: StorageDep,
    cross_referencer: CrossReferencerDep,
    scope: Annotated[
        str | None, Query(description="Restrict idea-seeds/crossrefs to a scope.")
    ] = None,
    entity: Annotated[
        str | None, Query(description="Center the subgraph on this entity.")
    ] = None,
    entity_type: Annotated[
        str | None, Query(description="Filter entity nodes by type.")
    ] = None,
    depth: Annotated[
        int,
        Query(
            ge=1,
            le=4,
            description="Neighborhood depth (hops); the bounded snapshot uses the one-hop set.",
        ),
    ] = 1,
    include_crossrefs: Annotated[
        bool, Query(description="Overlay cross-reference edges with their reasons (FR-RET-6).")
    ] = True,
    limit: Annotated[int, Query(ge=1, le=1000, description="Max nodes returned.")] = 200,
) -> GraphResponse:
    """A scoped entity/idea-seed subgraph for the explorer (PRD §4.4).

    The structural subgraph (entity nodes centered on ``entity`` when given, their
    weighted edges, and in-scope ``idea_seed`` memory nodes with their ``mentions``
    links) comes from the single bounded, embedding-free
    :meth:`StorageBackend.graph_snapshot` read. The cross-reference overlay
    (FR-RET-6) is a :class:`CrossReferencer` concern added on top as explained
    ``is_crossref`` edges. ``truncated`` is set when the node cap fires IN CYPHER.
    """

    scope_obj = scope_to_obj(scope)
    project = (
        scope_obj.project_id if scope_obj is not None and not scope_obj.is_global else None
    )

    # --- bounded structural snapshot (entities + edges + idea-seeds) ------
    snapshot = await storage.graph_snapshot(
        scope=scope_obj,
        entity=entity,
        entity_type=entity_type,
        include_idea_seeds=True,
        node_limit=limit,
    )
    nodes: list[GraphNode] = [_snapshot_node(n) for n in snapshot.nodes]
    edges: list[GraphEdge] = [_snapshot_edge(e) for e in snapshot.edges]
    truncated = snapshot.truncated

    # Canonical names of the kept entity nodes seed the cross-ref working context.
    entity_names = [n.label for n in snapshot.nodes if n.kind == "entity"]

    # --- cross-reference overlay (FR-RET-6) ------------------------------
    if include_crossrefs:
        overlay = await _crossref_overlay(
            cross_referencer, entity_names, scope, project
        )
        # Only keep overlay edges whose target entity node is in the subgraph (or
        # add a lightweight node for the source memory so the edge has an endpoint).
        node_ids = {n.id for n in nodes}
        crossref_mem_by_id: dict[str, str] = {}
        for cross_edge in overlay:
            if cross_edge.target not in node_ids:
                continue
            if cross_edge.source not in node_ids:
                # The cross-referenced memory may not be an idea-seed already in the
                # subgraph; add a compact node so the explained edge is anchored.
                crossref_mem_by_id[cross_edge.source] = cross_edge.source[len(_MEM_PREFIX) :]
            edges.append(cross_edge)
        for src_node_id, mem_id in crossref_mem_by_id.items():
            # Keyed, embedding-free read for just this cross-referenced memory.
            cross_mem = await storage.get_memory_display(mem_id)
            label = (
                " ".join(cross_mem.content.split())[:80]
                if cross_mem is not None
                else mem_id
            )
            nodes.append(
                GraphNode(
                    id=src_node_id,
                    label=label,
                    kind="memory",
                    scope=cross_mem.scope.as_str() if cross_mem is not None else None,
                    memory_count=1,
                )
            )
            node_ids.add(src_node_id)

    return GraphResponse(nodes=nodes, edges=edges, truncated=truncated)


__all__ = ["router"]
