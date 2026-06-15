"""Health + stats route tests (PRD §4.1) — offline against fakes.

Health must never fail the request (a down dependency is reported, not raised);
stats tally the live store. Against the in-memory fake the infra probes report
``down``/``degraded`` (no real Ollama/LLM/Cypher seam) and the request still 200s.

The /api/stats/growth tests pin the route's DENSIFICATION of the backend's sparse
``memory_growth`` series: a full days-length, zero-filled, oldest-first run with a
parallel cumulative running total and ``total == sum(daily) == cumulative[-1]``.
Memories are dated relative to the REAL wall-clock ``today`` (the route pins its
own UTC anchor and the schema does not expose a ``today`` override), so the
window math is asserted structurally + against fresh fixtures rather than the
pre-seeded fixed-date store.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from tests.conftest import InMemoryStorage


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


# ---------------------------------------------------------------------------
# /api/stats/growth — densified store-growth trend (Dashboard sparkline)
# ---------------------------------------------------------------------------


def _add_memory(
    storage: InMemoryStorage, *, mid: str, days_ago: int, scope: object | None = None
) -> None:
    """Add a memory whose valid_from is ``days_ago`` days before the real today.

    The route pins ``today = datetime.now(UTC).date()`` and the schema exposes no
    override, so growth-route tests must date memories relative to wall-clock now.
    """

    from mnemozine.schema.models import MemoryUnit, Provenance, Scope

    when = datetime.now(UTC) - timedelta(days=days_ago)
    storage.memories[mid] = MemoryUnit(
        id=mid,
        category="preference",
        content=f"growth memory {mid}",
        scope=scope if scope is not None else Scope.global_(),
        entities=["rust"],
        confidence=0.9,
        provenance=Provenance(source="claude_code", session_id="sess-g"),
        valid_from=when,
    )


def test_growth_densifies_to_full_window_zero_filled(
    client: TestClient, storage: InMemoryStorage
) -> None:
    """The sparse backend series is densified to a full, oldest-first day run.

    ``days`` is exactly the window length, oldest-first, ending at today; ``daily``
    is parallel + zero-filled on empty days; ``cumulative`` is the running total;
    and ``total == sum(daily) == cumulative[-1]`` (the windowed creation count).
    """

    # Two memories today, one 2 days ago; the pre-seeded fixture memories are all
    # >=3 days old (3..40), so a 4-day window picks up exactly these three + the
    # one pre-seeded memory at day 3 (mem-fact-tokio @ _NOW-3d is fixed-date and
    # likely outside the real window — so seed our own and assert structurally).
    _add_memory(storage, mid="g-today-a", days_ago=0)
    _add_memory(storage, mid="g-today-b", days_ago=0)
    _add_memory(storage, mid="g-2ago", days_ago=2)

    resp = client.get("/api/stats/growth", params={"days": 4})
    assert resp.status_code == 200
    body = resp.json()

    # Dense + parallel + oldest-first, length == window size.
    assert len(body["days"]) == 4
    assert len(body["daily"]) == 4
    assert len(body["cumulative"]) == 4
    assert body["days"] == sorted(body["days"])  # ascending day labels
    today = datetime.now(UTC).date().isoformat()
    assert body["days"][-1] == today  # window ends at today

    # Our three seeded memories land in this 4-day window: 2 today, 1 two days ago.
    assert body["daily"][-1] >= 2  # the two we added today (fixtures may add more)
    assert body["daily"][-3] >= 1  # the one two days ago

    # cumulative is the running total of daily; total == sum(daily) == cumulative[-1].
    running = 0
    for i, n in enumerate(body["daily"]):
        running += n
        assert body["cumulative"][i] == running
    assert body["total"] == sum(body["daily"]) == body["cumulative"][-1]


def test_growth_empty_window_is_all_zero(
    client: TestClient, storage: InMemoryStorage
) -> None:
    """No memory in the window -> an all-zero dense series, never a sparse gap.

    Drop the pre-seeded fixed-date fixtures so the trailing window is genuinely
    empty; the route must still return a full days-length zero-filled run (the
    "empty store yields an all-zero sparkline" contract — no activity-log message).
    """

    storage.memories.clear()

    body = client.get("/api/stats/growth", params={"days": 5}).json()
    assert body["days"] == sorted(body["days"]) and len(body["days"]) == 5
    assert body["daily"] == [0, 0, 0, 0, 0]
    assert body["cumulative"] == [0, 0, 0, 0, 0]
    assert body["total"] == 0


def test_growth_scope_filter_rolls_up_descendants(
    client: TestClient, storage: InMemoryStorage
) -> None:
    """A project scope rolls up its sub-scopes but never a sibling project."""

    from mnemozine.schema.models import Scope

    storage.memories.clear()
    _add_memory(storage, mid="g-mz", days_ago=1, scope=Scope.project("Mnemozine"))
    _add_memory(
        storage, mid="g-mz-auth", days_ago=1, scope=Scope.project("Mnemozine", "auth")
    )
    _add_memory(storage, mid="g-pulse", days_ago=1, scope=Scope.project("Pulse"))

    mz = client.get(
        "/api/stats/growth", params={"days": 7, "scope": "project:Mnemozine"}
    ).json()
    # Mnemozine + its auth sub-scope (2), never the Pulse sibling.
    assert mz["total"] == 2

    pulse = client.get(
        "/api/stats/growth", params={"days": 7, "scope": "project:Pulse"}
    ).json()
    assert pulse["total"] == 1

    # global rolls up the WHOLE store (all three).
    glob = client.get("/api/stats/growth", params={"days": 7, "scope": "global"}).json()
    assert glob["total"] == 3


def test_growth_rejects_out_of_range_days(client: TestClient) -> None:
    """The ``days`` query param is bounded (ge=1, le=365)."""

    assert client.get("/api/stats/growth", params={"days": 0}).status_code == 422
    assert client.get("/api/stats/growth", params={"days": 366}).status_code == 422
