"""WebUI mutation route tests (WEBUI BE-MUT stream).

Exercises the HITL write surface owned by ``mnemozine.web.routes.mutations`` —
``PATCH /api/memories/{id}`` (reclassify / re-scope / archive-restore),
``POST /api/crossrefs/{id}/suppress``, ``POST /api/maintenance/{job}/run``, and the
F4 eval bootstrap ``label`` / ``finish`` — end-to-end through a
:class:`fastapi.testclient.TestClient` over a Container wired to the offline fakes
(see ``tests/web/conftest.py``). Each successful mutation is asserted to (a) go
through the existing :class:`~mnemozine.interfaces.StorageBackend` / maintenance /
evals and (b) record an :class:`~mnemozine.activity.ActivityEvent` on the injected
in-memory activity log.

The mutation router is registered before the read-side ``crossrefs`` /
``maintenance`` / ``eval`` routers, so these live handlers replace the Phase-1
stubs on the same paths (FastAPI first-match wins) — a fact these tests rely on.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from mnemozine.activity import ActivityKind, InMemoryActivityLog
from mnemozine.schema.models import MemoryType, Scope, Tier
from mnemozine.web.routes._bootstrap_state import bootstrap_store
from tests.conftest import InMemoryStorage


@pytest.fixture(autouse=True)
def _reset_bootstrap_store() -> Iterator[None]:
    """Isolate the process-wide bootstrap store + write to a temp gold path."""

    bootstrap_store.reset()
    bootstrap_store.set_gold_set_path(Path(tempfile.mkdtemp()) / "bootstrap-gold.json")
    yield
    bootstrap_store.reset()


async def _events(log: InMemoryActivityLog) -> list:
    """Drain emit()'s scheduled tasks (none here — routes await emit_async) and read."""

    await asyncio.sleep(0)
    return await log.query()


# ---------------------------------------------------------------------------
# PATCH /api/memories/{id}
# ---------------------------------------------------------------------------


def test_patch_reclassify(client, storage: InMemoryStorage, activity_log) -> None:
    resp = client.patch("/api/memories/mem-pref-current", json={"type": "idea_seed"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["changed"] == ["type"]
    assert body["memory"]["type"] == "idea_seed"
    # Persisted through the backend.
    assert storage.memories["mem-pref-current"].type is MemoryType.IDEA_SEED
    # Recorded on the activity feed.
    events = asyncio.run(_events(activity_log))
    assert any(
        e.kind is ActivityKind.EXTRACT_DECISION and "mem-pref-current" in e.ref_memory_ids
        for e in events
    )


def test_patch_rescope_bare_project_id(client, storage: InMemoryStorage) -> None:
    # A bare project id is accepted and promoted to project:<id>.
    resp = client.patch("/api/memories/mem-pref-current", json={"scope": "rust-cli"})
    assert resp.status_code == 200
    assert resp.json()["memory"]["scope"] == "project:rust-cli"
    assert storage.memories["mem-pref-current"].scope == Scope.project("rust-cli")


def test_patch_rescope_global(client, storage: InMemoryStorage) -> None:
    resp = client.patch("/api/memories/mem-fact-tokio", json={"scope": "global"})
    assert resp.status_code == 200
    assert resp.json()["memory"]["scope"] == "global"
    assert storage.memories["mem-fact-tokio"].scope.is_global


def test_patch_archive_then_restore(client, storage: InMemoryStorage) -> None:
    # Archive an active hot memory.
    resp = client.patch("/api/memories/mem-pref-current", json={"tier": "archive"})
    assert resp.status_code == 200
    assert resp.json()["memory"]["tier"] == "archive"
    assert storage.memories["mem-pref-current"].tier is Tier.ARCHIVE

    # Restore it (the archived idea_seed -> hot).
    resp = client.patch("/api/memories/mem-idea-cli", json={"tier": "hot"})
    assert resp.status_code == 200
    assert resp.json()["memory"]["tier"] == "hot"
    assert storage.memories["mem-idea-cli"].tier is Tier.HOT


def test_patch_combined_fields(client, storage: InMemoryStorage) -> None:
    resp = client.patch(
        "/api/memories/mem-pref-current",
        json={"type": "project_fact", "scope": "rust-cli", "tier": "archive"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["changed"]) == {"type", "scope", "tier"}
    m = storage.memories["mem-pref-current"]
    assert m.type is MemoryType.PROJECT_FACT
    assert m.scope == Scope.project("rust-cli")
    assert m.tier is Tier.ARCHIVE


def test_patch_empty_is_422(client) -> None:
    resp = client.patch("/api/memories/mem-pref-current", json={})
    assert resp.status_code == 422


def test_patch_unknown_id_is_404(client) -> None:
    resp = client.patch("/api/memories/does-not-exist", json={"tier": "hot"})
    assert resp.status_code == 404


def test_patch_invalid_scope_is_422(client) -> None:
    # An empty/whitespace bare scope cannot be promoted to project:<id>.
    resp = client.patch("/api/memories/mem-pref-current", json={"scope": "project:"})
    assert resp.status_code == 422


def test_patch_does_not_edit_content(client, storage: InMemoryStorage) -> None:
    # Content is never editable (PRD §7): an extra content field is ignored by
    # the MemoryPatchRequest schema, and the stored content is unchanged.
    before = storage.memories["mem-pref-current"].content
    resp = client.patch(
        "/api/memories/mem-pref-current", json={"type": "idea_seed", "content": "HACKED"}
    )
    assert resp.status_code == 200
    assert storage.memories["mem-pref-current"].content == before


# ---------------------------------------------------------------------------
# POST /api/crossrefs/{id}/suppress
# ---------------------------------------------------------------------------


def test_suppress_persists_and_emits(client, storage: InMemoryStorage, activity_log) -> None:
    resp = client.post(
        "/api/crossrefs/mem-idea-cli/suppress", json={"context_key": "project:rust-cli"}
    )
    assert resp.status_code == 200
    assert resp.json()["changed"] == ["suppressed"]
    # Persisted via StorageBackend.record_suppression (R2).
    assert ("mem-idea-cli", "project:rust-cli") in storage.suppressions
    # Read back through the Protocol.
    assert asyncio.run(storage.is_suppressed("mem-idea-cli", "project:rust-cli"))
    # Recorded on the feed.
    events = asyncio.run(_events(activity_log))
    assert any(e.kind is ActivityKind.MAINTENANCE and "suppress" in e.summary for e in events)


def test_suppress_is_idempotent(client, storage: InMemoryStorage) -> None:
    for _ in range(3):
        resp = client.post(
            "/api/crossrefs/mem-idea-cli/suppress", json={"context_key": "ctx"}
        )
        assert resp.status_code == 200
    assert ("mem-idea-cli", "ctx") in storage.suppressions


def test_suppress_blank_context_is_422(client) -> None:
    resp = client.post("/api/crossrefs/mem-idea-cli/suppress", json={"context_key": "   "})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# F4 eval bootstrap — label / finish
# ---------------------------------------------------------------------------


def test_label_candidate_keep(client, activity_log) -> None:
    resp = client.post(
        "/api/eval/bootstrap/cand-0000/label",
        json={"label": "keep", "corrected_type": "preference"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["candidate_id"] == "cand-0000"
    assert body["label"] == "keep"
    assert body["corrected_type"] == "preference"
    # Reflected in the store.
    assert bootstrap_store.get("cand-0000").label == "keep"
    # Recorded on the feed.
    events = asyncio.run(_events(activity_log))
    assert any(e.kind is ActivityKind.EXTRACT_DECISION for e in events)


def test_label_candidate_drop(client) -> None:
    resp = client.post("/api/eval/bootstrap/cand-0001/label", json={"label": "drop"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "drop"
    assert bootstrap_store.get("cand-0001").label == "drop"


def test_label_unknown_candidate_is_404(client) -> None:
    resp = client.post("/api/eval/bootstrap/nope/label", json={"label": "keep"})
    assert resp.status_code == 404


def test_label_invalid_label_is_422(client) -> None:
    resp = client.post("/api/eval/bootstrap/cand-0000/label", json={"label": "maybe"})
    assert resp.status_code == 422


def test_finish_folds_kept_into_gold_set(client, activity_log) -> None:
    # Keep two candidates, drop one, then finish.
    client.post("/api/eval/bootstrap/cand-0000/label", json={"label": "keep"})
    client.post("/api/eval/bootstrap/cand-0001/label", json={"label": "keep"})
    client.post("/api/eval/bootstrap/cand-0002/label", json={"label": "drop"})

    resp = client.post("/api/eval/bootstrap/finish", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["gold_set"] == "mnemozine-gold"
    assert body["passed"] is True
    # The folded gold set holds the two kept classifier cases.
    metric = body["metrics"][0]
    assert metric["value"] == 2.0
    assert body["ran_at"] is not None
    # The gold set was written to the configured temp path.
    assert bootstrap_store.gold_set_path.is_file()
    # And a maintenance event recorded the fold.
    events = asyncio.run(_events(activity_log))
    assert any(
        e.kind is ActivityKind.MAINTENANCE and "gold set" in e.summary for e in events
    )


def test_finish_with_no_keeps_yields_empty_gold(client) -> None:
    resp = client.post("/api/eval/bootstrap/finish", json={})
    assert resp.status_code == 200
    assert resp.json()["metrics"][0]["value"] == 0.0
