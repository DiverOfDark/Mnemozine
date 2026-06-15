"""Memories list + detail routes (PRD §4.2 / §4.3, §6 GET /api/memories[/{id}]).

The core read surface. The list endpoint pushes the PRD §4.2 table filters — type
/ scope / tier / entity / active-vs-superseded / source / free-text — the
newest-first ordering, and the paging entirely into FalkorDB via
:meth:`StorageBackend.query_memories`, which returns embedding-free
:class:`~mnemozine.interfaces.MemoryView`s + a Cypher ``COUNT`` total (NEVER a
whole-store stream that loads/parses the 1024-float embedding per row). Detail
keys a single unit by id with :meth:`get_memory_display` (also embedding-free) and
reconstructs its **provenance + validity window + supersession chain** (the
signature temporal feature, PRD §2/§4.3) from a bounded same-scope/category
neighborhood (supersession is not a stored edge — it is the temporal adjacency a
closed validity window leaves behind, FR-MNT-1).

Bound to the live ``StorageBackend`` via the Container; runs identically against
the in-memory fake in tests. Read-only — the HITL writes live in ``mutations.py``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from mnemozine.schema.models import Tier
from mnemozine.web.deps import StorageDep
from mnemozine.web.routes._read import (
    build_scope_tree,
    scope_to_obj,
    view_to_detail,
    view_to_list_item,
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

    Every filter, the newest-first ordering (by ``valid_from``), and the
    ``offset``/``limit`` paging are pushed into FalkorDB via
    :meth:`StorageBackend.query_memories`; the embedding is projected out so it is
    never transferred for a list read, and ``total`` is a cheap Cypher ``COUNT`` of
    the whole filtered set (not a second whole-store scan). The route's query
    signature is identical to ``query_memories`` — a drop-in pass-through.
    """

    scope_obj = scope_to_obj(scope)
    page = await storage.query_memories(
        category=category,
        scope=scope_obj,
        tier=tier,
        entity=entity,
        source=source,
        active=active,
        q=q,
        limit=limit,
        offset=offset,
    )
    items = [view_to_list_item(v) for v in page.items]
    return MemoryListResponse(
        items=items, page=Page(total=page.total, limit=limit, offset=offset)
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

    return ScopeTreeResponse(root=build_scope_tree(await storage.scope_counts()))


@router.get("/{memory_id}", response_model=MemoryDetail, summary="Memory detail")
async def get_memory(memory_id: str, storage: StorageDep) -> MemoryDetail:
    """Full memory detail + provenance + validity + supersession chain (PRD §4.3).

    Keys the unit by id with the embedding-free
    :meth:`StorageBackend.get_memory_display` (404 when unknown), then derives the
    supersession chain against the unit's own same-scope/category neighborhood —
    fetched with :meth:`query_memories` (embedding-free, Cypher-paged), exactly the
    FR-MNT-1 comparison set the supersede write used, never a whole-store stream.
    """

    target = await storage.get_memory_display(memory_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="memory not found"
        )
    # Derive the chain against the same-scope/category neighborhood only (cheaper +
    # exactly the FR-MNT-1 comparison set the supersede write used). Bounded by the
    # route's max page size; the entity-overlapping same-category set is small.
    neighborhood = await storage.query_memories(
        scope=target.scope, category=target.category, limit=500, offset=0
    )
    return view_to_detail(target, neighborhood.items)


__all__ = ["router"]
