"""Activity / Logs route (PRD §4.6 / §6 GET /api/activity).

This route is wired to the real :class:`~mnemozine.interfaces.ActivityLog` from
the Container — but the default log is :class:`NullActivityLog`, so unless
``web.enable_activity_log`` is set it simply returns an empty feed (schema-valid).
That keeps the stub honest: the contract and the query plumbing are real, and the
moment the persisted log is enabled this route serves live data with no change.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query

from mnemozine.activity.models import ActivityKind, ActivityQuery
from mnemozine.web.deps import ActivityLogDep
from mnemozine.web.schemas import (
    ActivityEventOut,
    ActivityResponse,
    Page,
)

router = APIRouter(prefix="/api/activity", tags=["activity"])


@router.get("", response_model=ActivityResponse, summary="Chronological activity feed")
async def list_activity(
    log: ActivityLogDep,
    kind: Annotated[
        list[ActivityKind] | None,
        Query(description="Filter by event kind(s): ingest/extract_decision/maintenance/injection"),
    ] = None,
    source: Annotated[str | None, Query(description="Filter by originating source.")] = None,
    session_id: Annotated[str | None, Query(description="Filter by session id.")] = None,
    project: Annotated[str | None, Query(description="Filter by project.")] = None,
    ref_memory_id: Annotated[
        str | None, Query(description="Only events that touched this memory id.")
    ] = None,
    since: Annotated[datetime | None, Query(description="Inclusive lower time bound.")] = None,
    until: Annotated[datetime | None, Query(description="Exclusive upper time bound.")] = None,
    limit: Annotated[int, Query(ge=1, le=1000, description="Page size.")] = 100,
    offset: Annotated[int, Query(ge=0, description="Page offset.")] = 0,
) -> ActivityResponse:
    """The chronological, filterable activity feed (PRD §4.6).

    Reads the live activity log (empty under the default NullActivityLog; live
    once ``web.enable_activity_log`` is on). Newest-first, paged.
    """

    query = ActivityQuery(
        kinds=kind,
        source=source,
        session_id=session_id,
        project=project,
        ref_memory_id=ref_memory_id,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )
    events = await log.query(query)
    items = [ActivityEventOut.model_validate(e.model_dump()) for e in events]
    return ActivityResponse(
        items=items,
        page=Page(total=len(items), limit=limit, offset=offset),
    )


__all__ = ["router"]
