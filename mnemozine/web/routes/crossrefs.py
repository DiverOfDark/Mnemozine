"""Cross-reference read route (PRD §4.4/§4.7 / §6 GET /api/crossrefs).

Lists the surfaced serendipitous connections for a working context — each with
its mandatory human-readable ``reason`` (FR-RET-6), shared entities, and the
``context_key`` a suppression would apply to — by running the live
``CrossReferencer.find_related``.

The suppress *write* (``POST /api/crossrefs/{id}/suppress``) lives in
``mutations.py`` (the single auditable write surface, PRD §2); this module is
read-only. ``find_related`` already excludes suppressed connections, so the list
is the active suggestions. Runs against the in-memory fake in tests.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from mnemozine.crossref.engine import context_key_for
from mnemozine.interfaces import RetrievalContext
from mnemozine.web.deps import CrossReferencerDep
from mnemozine.web.routes._read import memory_to_list_item, scope_to_obj
from mnemozine.web.schemas import (
    CrossRefItem,
    CrossRefResponse,
    Page,
)

router = APIRouter(prefix="/api/crossrefs", tags=["crossrefs"])


@router.get("", response_model=CrossRefResponse, summary="Cross-references for a context")
async def list_crossrefs(
    cross_referencer: CrossReferencerDep,
    project: Annotated[str | None, Query(description="Working-context project.")] = None,
    entity: Annotated[str | None, Query(description="Filter by a shared entity.")] = None,
    include_suppressed: Annotated[
        bool, Query(description="Include dismissed/suppressed connections (R2).")
    ] = False,
    limit: Annotated[int, Query(ge=1, le=200, description="Page size.")] = 50,
    offset: Annotated[int, Query(ge=0, description="Page offset.")] = 0,
) -> CrossRefResponse:
    """Surfaced serendipitous connections for a context (PRD §4.4 / FR-RET-6).

    Builds a working context from ``project`` + ``entity`` and runs the live
    cross-referencer. Every item carries its non-empty reason, shared entities, and
    the ``context_key`` a dismissal applies to. (``find_related`` already drops
    suppressed connections; ``include_suppressed`` is accepted for the contract but
    the active list is what the engine returns.)
    """

    entities = [entity] if entity else []
    scope_obj = scope_to_obj(project)
    project_id = (
        scope_obj.project_id if scope_obj is not None and not scope_obj.is_global else None
    )
    context = RetrievalContext(project=project_id, entities=entities)
    ctx_key = context_key_for(context)

    # Pull a wide page so offset/limit can slice without re-running the engine.
    refs = await cross_referencer.find_related(context, max_suggestions=offset + limit)

    items = [
        CrossRefItem(
            memory=memory_to_list_item(ref.memory),
            score=ref.score,
            reason=ref.reason,
            shared_entities=list(ref.shared_entities),
            suppressed=False,
            context_key=ctx_key,
        )
        for ref in refs
    ]
    total = len(items)
    page_items = items[offset : offset + limit]
    return CrossRefResponse(
        items=page_items, page=Page(total=total, limit=limit, offset=offset)
    )


__all__ = ["router"]
