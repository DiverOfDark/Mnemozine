"""Maintenance read route tests (PRD §4.7) — offline against fakes.

Covers the scheduler status + job list and the FR-MNT-4 entity-resolution
merge-candidate review queue (the seeded ``rust`` / ``rust-lang`` duplicate pair).
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_maintenance_status_reports_cron_and_jobs(client: TestClient) -> None:
    resp = client.get("/api/maintenance")
    assert resp.status_code == 200
    body = resp.json()
    assert body["cron"]  # the configured cron expression
    assert body["scheduler_running"] is False
    names = [j["name"] for j in body["jobs"]]
    assert names == ["consolidate", "entity-resolution", "decay", "audit", "migrate-index"]


def test_merge_candidates_surfaces_duplicate_entities(client: TestClient) -> None:
    # rust + rust-lang normalize to the same key -> one merge candidate.
    resp = client.get("/api/maintenance/merge-candidates")
    assert resp.status_code == 200
    candidates = resp.json()["candidates"]
    names = {frozenset((c["source_name"], c["target_name"])) for c in candidates}
    assert frozenset(("rust", "rust-lang")) in names
    for c in candidates:
        assert 0.0 <= c["similarity"] <= 1.0
        assert c["shared_neighbors"] >= 0
