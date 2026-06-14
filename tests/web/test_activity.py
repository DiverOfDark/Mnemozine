"""Activity / Logs route tests (PRD §4.6) — offline against the in-memory log.

The /api/activity route reads the live ActivityLog. Under the default
NullActivityLog the feed is empty by design; with the InMemoryActivityLog wired
(as the test container does) appended events are queryable and filterable.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from mnemozine.activity.log import InMemoryActivityLog
from mnemozine.activity.models import ingest_event, maintenance_event


async def _append(log: InMemoryActivityLog) -> None:
    await log.append(
        ingest_event(
            source="claude_code",
            session_id="sess-1",
            project="rust-cli",
            summary="ingested chunk",
            ref_memory_ids=["mem-pref-current"],
        )
    )
    await log.append(
        maintenance_event(job_name="decay", summary="decay sweep ran")
    )


def test_activity_empty_feed_is_schema_valid(client: TestClient) -> None:
    body = client.get("/api/activity").json()
    assert body["items"] == []
    assert body["page"]["total"] == 0


async def test_activity_returns_appended_events(
    client: TestClient, activity_log: InMemoryActivityLog
) -> None:
    await _append(activity_log)
    body = client.get("/api/activity").json()
    kinds = {e["kind"] for e in body["items"]}
    assert kinds == {"ingest", "maintenance"}


async def test_activity_filter_by_kind(
    client: TestClient, activity_log: InMemoryActivityLog
) -> None:
    await _append(activity_log)
    body = client.get("/api/activity", params={"kind": "ingest"}).json()
    assert {e["kind"] for e in body["items"]} == {"ingest"}
    assert body["items"][0]["ref_memory_ids"] == ["mem-pref-current"]


async def test_activity_filter_by_ref_memory_id(
    client: TestClient, activity_log: InMemoryActivityLog
) -> None:
    await _append(activity_log)
    body = client.get(
        "/api/activity", params={"ref_memory_id": "mem-pref-current"}
    ).json()
    assert len(body["items"]) == 1
    assert body["items"][0]["kind"] == "ingest"
