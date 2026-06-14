"""Memories list + detail routes (PRD §4.2 / §4.3, §6 GET /api/memories[/{id}]).

The core read surface. The list endpoint streams the store
(:meth:`StorageBackend.iter_memories`, the Protocol's only whole-store entry
point) and applies the PRD §4.2 table filters — type / scope / tier / entity /
active-vs-superseded / source / free-text — in Python, then pages. Detail keys a
single unit by id and reconstructs its **provenance + validity window +
supersession chain** (the signature temporal feature, PRD §2/§4.3) from the
same-scope/entity neighborhood (supersession is not a stored edge — it is the
temporal adjacency a closed validity window leaves behind, FR-MNT-1).

Bound to the live ``StorageBackend`` via the Container; runs identically against
the in-memory fake in tests. Read-only — the HITL writes live in ``mutations.py``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from mnemozine.schema.models import MemoryUnit, Tier
from mnemozine.web.deps import StorageDep
from mnemozine.web.routes._read import (
    build_scope_tree,
    collect_memories,
    memory_to_detail,
    memory_to_list_item,
    scope_to_obj,
)
from mnemozine.web.schemas import (
    CategoryFacet,
    CategoryFacetsResponse,
    MemoryDetail,
    MemoryListResponse,
    Page,
    ScopeTreeResponse,
)

router = APIRouter(prefix="/api/memories", tags=["memories"])


def _matches(
    memory: MemoryUnit,
    *,
    category: str | None,
    tier: Tier | None,
    entity: str | None,
    source: str | None,
    active: bool | None,
    q: str | None,
) -> bool:
    """Apply the in-memory table filters to one unit (PRD §4.2)."""

    if category is not None and memory.category != category.strip().lower():
        return False
    if tier is not None and memory.tier != tier:
        return False
    if active is not None and memory.is_active != active:
        return False
    if source is not None and memory.provenance.source != source:
        return False
    if entity is not None:
        wanted = entity.lower()
        if wanted not in {e.lower() for e in memory.entities}:
            return False
    if q:
        if q.lower() not in memory.content.lower():
            return False
    return True


@router.get("", response_model=MemoryListResponse, summary="List/filter memories")
async def list_memories(
    storage: StorageDep,
    category: Annotated[
        str | None, Query(description="Filter by free-form category.")
    ] = None,
    scope: Annotated[
        str | None, Query(description="Filter by scope ('global'/'project:<id>'/bare id).")
    ] = None,
    tier: Annotated[Tier | None, Query(description="Filter by tier (hot/archive).")] = None,
    entity: Annotated[str | None, Query(description="Filter by linked entity name.")] = None,
    source: Annotated[str | None, Query(description="Filter by ingest source.")] = None,
    active: Annotated[
        bool | None, Query(description="True=active only, False=superseded only, None=both.")
    ] = None,
    q: Annotated[str | None, Query(description="Free-text content search.")] = None,
    limit: Annotated[int, Query(ge=1, le=500, description="Page size.")] = 50,
    offset: Annotated[int, Query(ge=0, description="Page offset.")] = 0,
) -> MemoryListResponse:
    """Filterable, paged Memories table (PRD §4.2).

    Scope filtering is pushed into ``iter_memories`` (the backend bounds it); the
    remaining filters are applied in Python over the streamed units. Results are
    ordered newest-first by ``valid_from`` so the freshest facts lead the table,
    then sliced by ``offset``/``limit`` after computing the unfiltered total.
    """

    scope_obj = scope_to_obj(scope)
    units = await collect_memories(storage, scope=scope_obj)
    matched = [
        m
        for m in units
        if _matches(
            m,
            category=category,
            tier=tier,
            entity=entity,
            source=source,
            active=active,
            q=q,
        )
    ]
    matched.sort(key=lambda m: m.valid_from, reverse=True)
    total = len(matched)
    page = matched[offset : offset + limit]
    items = [memory_to_list_item(m) for m in page]
    return MemoryListResponse(
        items=items, page=Page(total=total, limit=limit, offset=offset)
    )


@router.get(
    "/facets/categories",
    response_model=CategoryFacetsResponse,
    summary="Discovered category facets",
)
async def category_facets(storage: StorageDep) -> CategoryFacetsResponse:
    """Distinct free-form categories in use + their counts (dynamic facet).

    Categories are open-ended now (no fixed enum), so the UI discovers the filter
    chips from the store. Reads :meth:`StorageBackend.list_categories` (the
    contract's category registry: ``(category, active-count)`` pairs) and returns
    them ordered most-frequent first so the busiest categories lead the facet.
    """

    pairs = await storage.list_categories()
    facets = [CategoryFacet(category=cat, count=count) for cat, count in pairs]
    facets.sort(key=lambda f: (-f.count, f.category))
    return CategoryFacetsResponse(facets=facets, total=len(facets))


@router.get(
    "/facets/scope-tree",
    response_model=ScopeTreeResponse,
    summary="Hierarchical scope tree with counts",
)
async def scope_tree(storage: StorageDep) -> ScopeTreeResponse:
    """The project/sub-scope hierarchy with per-node counts (scope navigator).

    Streams the whole store and folds every memory's hierarchical
    :class:`~mnemozine.schema.models.Scope` path into a tree rooted at ``global``.
    Each node carries its exact ``count`` and a ``total_count`` rolled up over its
    descendants — exactly the ancestor-composed view selecting that scope yields
    from its subtree (no-leak: a node never counts a sibling subtree).
    """

    units = await collect_memories(storage)
    return ScopeTreeResponse(root=build_scope_tree(units))


@router.get("/{memory_id}", response_model=MemoryDetail, summary="Memory detail")
async def get_memory(memory_id: str, storage: StorageDep) -> MemoryDetail:
    """Full memory detail + provenance + validity + supersession chain (PRD §4.3).

    Streams the store to key the unit by id (the Protocol has no key-read), then
    derives the supersession chain against the unit's own scope neighborhood. 404
    when the id is unknown.
    """

    units = await collect_memories(storage)
    target = next((m for m in units if m.id == memory_id), None)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="memory not found"
        )
    # Derive the chain against the same-scope neighborhood only (cheaper + exactly
    # the FR-MNT-1 comparison set the supersede write used).
    scope = target.scope.as_str()
    universe = [m for m in units if m.scope.as_str() == scope]
    return memory_to_detail(target, universe)


__all__ = ["router"]
