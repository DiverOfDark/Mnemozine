"""WebUI maintenance-trigger mutation tests (POST /api/maintenance/{job}/run).

The operator-triggered maintenance run is a write (it mutates the store) so it
lives in ``mnemozine.web.routes.mutations`` and is covered here. It runs the
requested job through the real
:class:`~mnemozine.maintenance.runner.build_default_jobs` over the container's
offline-faked storage/LLM/embedding providers, returns the job's
:class:`MaintenanceReport`, and records a ``maintenance`` activity event. The
contract job names (``consolidate`` / ``entity-resolution`` / ``decay`` /
``audit``) are mapped onto the internal ``MaintenanceJob.name`` values.
"""

from __future__ import annotations

import asyncio

import pytest

from mnemozine.activity import ActivityKind, InMemoryActivityLog


def _drain_and_query(log: InMemoryActivityLog) -> list:
    async def _run() -> list:
        await asyncio.sleep(0)
        return await log.query()

    return asyncio.run(_run())


@pytest.mark.parametrize("job", ["consolidate", "entity-resolution", "decay", "audit"])
def test_run_job_succeeds_and_emits(client, activity_log: InMemoryActivityLog, job: str) -> None:
    resp = client.post(f"/api/maintenance/{job}/run", json={})
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert body["job"] == job
    assert body["started"] is True
    # A real MaintenanceReport is returned (offline jobs run clean over the fakes).
    assert body["report"] is not None
    assert isinstance(body["report"]["consolidated"], int)
    # The run is recorded on the activity feed.
    events = _drain_and_query(activity_log)
    assert any(
        e.kind is ActivityKind.MAINTENANCE and e.detail.get("triggered_by") == "operator"
        for e in events
    )


def test_run_job_unknown_is_404(client) -> None:
    resp = client.post("/api/maintenance/not-a-job/run", json={})
    assert resp.status_code == 404
    assert "unknown maintenance job" in resp.json()["detail"]


def test_run_job_report_shape(client) -> None:
    resp = client.post("/api/maintenance/decay/run", json={})
    assert resp.status_code == 200
    report = resp.json()["report"]
    # MaintenanceReportOut wire shape.
    for key in ("job_name", "consolidated", "entities_merged", "archived", "edges_pruned", "notes"):
        assert key in report
    assert isinstance(report["notes"], list)
