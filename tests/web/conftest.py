"""Offline fixtures for the WebUI READ-route tests (WEBUI BE-READ stream).

The read routes go through the live :class:`~mnemozine.app.Container` /
``StorageBackend`` / retriever / cross-referencer / activity log — never a new
source of truth (WEBUI PRD §2). These fixtures build a :class:`Container` with the
conftest **fakes** pre-injected onto its memoized slots, so the whole API is
exercised through a :class:`fastapi.testclient.TestClient` with **no FalkorDB /
Ollama / Qwen**:

* ``_storage``   — :class:`tests.conftest.InMemoryStorage` (real 4-way write + scan)
* ``_embedding`` — :class:`tests.conftest.FakeEmbeddingProvider`
* ``_llm``       — :class:`tests.conftest.FakeLLMProvider`
* ``_activity``  — :class:`~mnemozine.activity.log.InMemoryActivityLog`

The SPA static mount is disabled (``web.static_dir`` -> a non-existent path) so
``create_app`` builds API-only; the read routes are what these tests cover.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mnemozine.activity.log import InMemoryActivityLog
from mnemozine.app import Container
from mnemozine.config import Settings
from mnemozine.schema.models import (
    Edge,
    Entity,
    MemoryType,
    MemoryUnit,
    Provenance,
    Scope,
    Tier,
)
from mnemozine.web import create_app
from tests.conftest import FakeEmbeddingProvider, FakeLLMProvider, InMemoryStorage

_NOW = datetime(2026, 6, 14, 12, 0, 0, tzinfo=UTC)


def _prov(source: str = "claude_code", session: str = "sess-1") -> Provenance:
    return Provenance(
        source=source,
        session_id=session,
        chunk_hash="deadbeef",
        raw_path=f"~/.claude/projects/demo/{session}.jsonl",
    )


def seed_storage(storage: InMemoryStorage) -> None:
    """Seed a representative graph: memories (active/superseded/archived), entities, edges.

    Covers every filter axis the Memories table exposes (type / scope / tier /
    active-vs-superseded / source) plus a supersession pair, entities with a
    duplicate group (for merge-candidates), and weighted edges (for the graph).
    """

    # --- entities -------------------------------------------------------
    rust = Entity(id="ent-rust", canonical_name="rust", type="language")
    err = Entity(id="ent-err", canonical_name="error-handling", type="concept")
    tokio = Entity(id="ent-tokio", canonical_name="tokio", type="library")
    # A near-duplicate of rust for the entity-resolution merge-candidate queue.
    rust_lang = Entity(id="ent-rustlang", canonical_name="rust-lang", type="language")
    for e in (rust, err, tokio, rust_lang):
        storage.entities[e.id] = e

    # --- edges (weighted, one below the prune floor for variety) --------
    storage.edges["edge-rust-err"] = Edge(
        id="edge-rust-err",
        from_entity="ent-rust",
        to_entity="ent-err",
        relation="relates_to",
        weight=0.8,
    )
    storage.edges["edge-rust-tokio"] = Edge(
        id="edge-rust-tokio",
        from_entity="ent-rust",
        to_entity="ent-tokio",
        relation="uses",
        weight=0.5,
    )

    # --- memories -------------------------------------------------------
    # 1) active global preference (current value of a superseded pair).
    storage.memories["mem-pref-current"] = MemoryUnit(
        id="mem-pref-current",
        type=MemoryType.PREFERENCE,
        content="Prefers thiserror over anyhow for Rust error handling.",
        scope=Scope.global_(),
        entities=["rust", "error-handling"],
        confidence=0.95,
        provenance=_prov(),
        valid_from=_NOW - timedelta(days=5),
        last_accessed=_NOW - timedelta(hours=3),
        access_count=7,
    )
    # 2) the stale, superseded preference it replaced (closed validity window).
    stale = MemoryUnit(
        id="mem-pref-stale",
        type=MemoryType.PREFERENCE,
        content="Prefers anyhow over thiserror for Rust error handling.",
        scope=Scope.global_(),
        entities=["rust", "error-handling"],
        confidence=0.8,
        provenance=_prov(),
        valid_from=_NOW - timedelta(days=40),
    )
    stale.supersede(at=_NOW - timedelta(days=5))
    storage.memories["mem-pref-stale"] = stale
    # 3) a project_fact (different scope + source) for scope/source filters.
    storage.memories["mem-fact-tokio"] = MemoryUnit(
        id="mem-fact-tokio",
        type=MemoryType.PROJECT_FACT,
        content="The rust-cli project pins tokio 1.38.",
        scope=Scope.project("rust-cli"),
        entities=["tokio", "rust"],
        confidence=0.9,
        provenance=_prov(source="openai", session="sess-2"),
        valid_from=_NOW - timedelta(days=3),
    )
    # 4) an archived idea_seed (tier=archive) for tier filter + graph idea node.
    storage.memories["mem-idea-cli"] = MemoryUnit(
        id="mem-idea-cli",
        type=MemoryType.IDEA_SEED,
        content="Idea: an async CLI that streams logs with a tokio runtime.",
        scope=Scope.global_(),
        entities=["tokio", "rust"],
        confidence=0.7,
        provenance=_prov(source="hermes", session="sess-3"),
        tier=Tier.ARCHIVE,
        valid_from=_NOW - timedelta(days=30),
    )


@pytest.fixture
def storage() -> InMemoryStorage:
    s = InMemoryStorage()
    seed_storage(s)
    return s


@pytest.fixture
def activity_log() -> InMemoryActivityLog:
    return InMemoryActivityLog()


@pytest.fixture
def container(
    storage: InMemoryStorage, activity_log: InMemoryActivityLog
) -> Container:
    """A Container with all layers pre-wired to offline fakes (no network)."""

    settings = Settings()
    # Disable the SPA static mount so create_app builds API-only (see module doc).
    settings.web.static_dir = Path("/nonexistent-spa-dir-for-tests")
    c = Container(settings=settings)
    c._storage = storage
    c._embedding = FakeEmbeddingProvider()
    c._llm = FakeLLMProvider()
    c._activity = activity_log
    return c


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    """A FastAPI TestClient over the offline-wired app."""

    app = create_app(container)
    with TestClient(app) as test_client:
        yield test_client
