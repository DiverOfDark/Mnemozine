"""Graph explorer route (PRD §4.4 / §6 GET /api/graph).

Builds a scoped subgraph for the Cytoscape view from the live store: entity nodes
(:meth:`StorageBackend.iter_entities`) joined by their weighted, temporal
relationship edges (:meth:`edges_for_entity`), plus ``idea_seed`` memory nodes
(first-class graph nodes that power cross-referencing, §7) linked to the entities
they mention. Cross-reference connections (FR-RET-6) are overlaid as ``is_crossref``
edges carrying their mandatory human-readable ``reason`` so the UI can highlight
serendipitous links distinctly from structural ones.

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
    RetrievalContext,
    StorageBackend,
)
from mnemozine.schema.models import Edge, Entity, MemoryUnit
from mnemozine.web.deps import CrossReferencerDep, StorageDep
from mnemozine.web.routes._read import collect_memories, scope_to_obj
from mnemozine.web.schemas import GraphEdge, GraphNode, GraphResponse

router = APIRouter(prefix="/api/graph", tags=["graph"])

# Stable node-id prefixes so entity and memory nodes never collide in Cytoscape.
_ENT_PREFIX = "ent:"
_MEM_PREFIX = "mem:"


def _entity_node(entity: Entity, memory_count: int) -> GraphNode:
    return GraphNode(
        id=f"{_ENT_PREFIX}{entity.id}",
        label=entity.canonical_name,
        kind="entity",
        entity_type=entity.type,
        memory_count=memory_count,
    )


def _idea_seed_node(memory: MemoryUnit) -> GraphNode:
    snippet = " ".join(memory.content.split())
    if len(snippet) > 80:
        snippet = snippet[:79].rstrip() + "…"
    return GraphNode(
        id=f"{_MEM_PREFIX}{memory.id}",
        label=snippet,
        kind="idea_seed",
        scope=memory.scope.as_str(),
        memory_count=1,
    )


def _structural_edge(edge: Edge) -> GraphEdge:
    return GraphEdge(
        id=edge.id,
        source=f"{_ENT_PREFIX}{edge.from_entity}",
        target=f"{_ENT_PREFIX}{edge.to_entity}",
        relation=edge.relation,
        weight=edge.weight,
        active=edge.is_active,
        is_crossref=False,
        reason=None,
    )


async def _collect_entities(
    storage: StorageBackend, *, entity_type: str | None
) -> list[Entity]:
    """Stream entity nodes, optionally filtered by entity type."""

    out: list[Entity] = []
    async for ent in storage.iter_entities():
        if entity_type is not None and (ent.type or "") != entity_type:
            continue
        out.append(ent)
    return out


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
        int, Query(ge=1, le=4, description="Neighborhood traversal depth (hops).")
    ] = 1,
    include_crossrefs: Annotated[
        bool, Query(description="Overlay cross-reference edges with their reasons (FR-RET-6).")
    ] = True,
    limit: Annotated[int, Query(ge=1, le=1000, description="Max nodes returned.")] = 200,
) -> GraphResponse:
    """A scoped entity/idea-seed subgraph for the explorer (PRD §4.4).

    Entity nodes come from the store (centered on ``entity`` when given), joined by
    their weighted edges; ``idea_seed`` memories in scope are added as nodes linked
    to the entities they mention; the cross-reference overlay (FR-RET-6) adds
    explained ``is_crossref`` edges. ``truncated`` is set when the node cap fires.
    """

    scope_obj = scope_to_obj(scope)
    project = (
        scope_obj.project_id if scope_obj is not None and not scope_obj.is_global else None
    )

    # --- entity nodes + structural edges ---------------------------------
    entities = await _collect_entities(storage, entity_type=entity_type)
    if entity is not None:
        # Center on one entity: keep it + its (up to `depth`-hop) neighbors.
        center = await storage.get_entity(entity)
        keep_ids: set[str] = set()
        if center is not None:
            keep_ids.add(center.id)
            frontier = [center.canonical_name]
            seen = {center.canonical_name}
            for _ in range(depth):
                nxt: list[str] = []
                for name in frontier:
                    for nb in await storage.neighbors(name, active_only=False):
                        keep_ids.add(nb.entity.id)
                        if nb.entity.canonical_name not in seen:
                            seen.add(nb.entity.canonical_name)
                            nxt.append(nb.entity.canonical_name)
                frontier = nxt
                if not frontier:
                    break
        entities = [e for e in entities if e.id in keep_ids]

    entity_by_id = {e.id: e for e in entities}
    entity_id_by_name = {e.canonical_name.lower(): e.id for e in entities}

    # Memories in scope, to count per-entity links and to surface idea-seeds.
    memories = await collect_memories(storage, scope=scope_obj)
    memory_count: dict[str, int] = {}
    for mem in memories:
        for name in mem.entities:
            eid = entity_id_by_name.get(name.lower())
            if eid is not None:
                memory_count[eid] = memory_count.get(eid, 0) + 1

    nodes: list[GraphNode] = [
        _entity_node(e, memory_count.get(e.id, 0)) for e in entities
    ]

    # Structural edges between kept entities (active + closed; UI greys closed).
    edges: list[GraphEdge] = []
    seen_edges: set[str] = set()
    for ent in entities:
        for edge in await storage.edges_for_entity(ent.canonical_name, active_only=False):
            if edge.id in seen_edges:
                continue
            if edge.from_entity in entity_by_id and edge.to_entity in entity_by_id:
                seen_edges.add(edge.id)
                edges.append(_structural_edge(edge))

    # --- cross-reference seed memory nodes -------------------------------
    # The old idea_seed type is now the cross_ref_candidate flag (FR-RET-6): these
    # are the first-class memory nodes that power serendipitous cross-references.
    idea_seeds = [m for m in memories if m.cross_ref_candidate]
    for seed in idea_seeds:
        nodes.append(_idea_seed_node(seed))
        for name in seed.entities:
            eid = entity_id_by_name.get(name.lower())
            if eid is None:
                continue
            edges.append(
                GraphEdge(
                    id=f"mentions:{seed.id}:{eid}",
                    source=f"{_MEM_PREFIX}{seed.id}",
                    target=f"{_ENT_PREFIX}{eid}",
                    relation="mentions",
                    weight=1.0,
                    active=seed.is_active,
                    is_crossref=False,
                    reason=None,
                )
            )

    # --- cross-reference overlay (FR-RET-6) ------------------------------
    if include_crossrefs:
        entity_names = [e.canonical_name for e in entities]
        overlay = await _crossref_overlay(
            cross_referencer, entity_names, scope, project
        )
        # Only keep overlay edges whose source memory node is in the subgraph (or
        # add a lightweight node for it so the highlighted edge has an endpoint).
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
        memory_by_id = {m.id: m for m in memories}
        for src_node_id, mem_id in crossref_mem_by_id.items():
            cross_mem = memory_by_id.get(mem_id)
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

    # --- node cap / truncation -------------------------------------------
    truncated = len(nodes) > limit
    if truncated:
        kept = nodes[:limit]
        kept_ids = {n.id for n in kept}
        nodes = kept
        edges = [e for e in edges if e.source in kept_ids and e.target in kept_ids]

    return GraphResponse(nodes=nodes, edges=edges, truncated=truncated)


__all__ = ["router"]
