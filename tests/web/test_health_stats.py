"""Health + stats route tests (PRD §4.1) — offline against fakes.

Health must never fail the request (a down dependency is reported, not raised);
stats tally the live store. Against the in-memory fake the infra probes report
``down``/``degraded`` (no real Ollama/LLM/Cypher seam) and the request still 200s.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_never_500s_and_reports_components(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"]
    assert body["status"] in {"ok", "degraded", "down"}
    names = {c["name"] for c in body["components"]}
    assert names == {"falkordb", "ollama", "llm"}
    assert body["activity_log_enabled"] is False


def test_health_falkordb_in_memory_backend_ok(client: TestClient) -> None:
    # The in-memory fake has no Cypher seam: falkordb reports ok with a note.
    components = {c["name"]: c for c in client.get("/api/health").json()["components"]}
    assert components["falkordb"]["status"] == "ok"


def test_stats_counts_by_category_scope_tier_source(client: TestClient) -> None:
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_memories"] == 4
    # Free-form categories replace the old fixed type enum (counts over all rows,
    # so the superseded 'preference' is included here).
    assert body["by_category"] == {"preference": 2, "decision": 1, "idea": 1}
    # The controlled scope decision is derived from the hierarchical scope.
    assert body["by_scope_decision"] == {"global": 3, "project": 1}
    assert body["by_tier"] == {"hot": 3, "archive": 1}
    assert body["by_source"] == {"claude_code": 2, "openai": 1, "hermes": 1}


def test_stats_active_vs_superseded_and_entities(client: TestClient) -> None:
    body = client.get("/api/stats").json()
    assert body["active_count"] == 3
    assert body["superseded_count"] == 1
    assert body["entity_count"] == 4
