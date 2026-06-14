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

from mnemozine.schema.models import MemoryType, MemoryUnit, Tier
from mnemozine.web.deps import StorageDep
from mnemozine.web.routes._read import (
    collect_memories,
    memory_to_detail,
    memory_to_list_item,
    scope_to_obj,
)
from mnemozine.web.schemas import (
    MemoryDetail,
    MemoryListResponse,
    Page,
)

router = APIRouter(prefix="/api/memories", tags=["memories"])


def _matches(
    memory: MemoryUnit,
    *,
    type_: MemoryType | None,
    tier: Tier | None,
    entity: str | None,
    source: str | None,
    active: bool | None,
    q: str | None,
) -> bool:
    """Apply the in-memory table filters to one unit (PRD §4.2)."""

    if type_ is not None and memory.type != type_:
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
    type: Annotated[MemoryType | None, Query(description="Filter by memory type.")] = None,
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
            type_=type,
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
