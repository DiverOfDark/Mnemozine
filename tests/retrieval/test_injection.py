"""FR-RET-3 / FR-RET-5 injection helper tests against the fakes."""

from __future__ import annotations

from mnemozine.config import Settings
from mnemozine.retrieval.injection import (
    mid_session_injection,
    session_start_injection,
)
from mnemozine.retrieval.retriever import ScopedRetriever
from mnemozine.schema.models import (
    MemoryUnit,
    Provenance,
    Scope,
)
from tests.conftest import InMemoryStorage


def _mem(
    content: str,
    scope: Scope,
    entities: list[str],
    *,
    category: str = "fact",
    cross_ref_candidate: bool = False,
) -> MemoryUnit:
    """Build a MemoryUnit on the category-split contract (no legacy ``type``)."""

    return MemoryUnit(
        content=content,
        scope=scope,
        category=category,
        cross_ref_candidate=cross_ref_candidate,
        entities=entities,
        confidence=0.9,
        provenance=Provenance(source="claude_code", session_id="s"),
    )


async def test_session_start_injection_under_budget(tmp_path, settings: Settings) -> None:
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "rust-cli"\n[dependencies]\nthiserror = "1"\n',
        encoding="utf-8",
    )
    storage = InMemoryStorage()
    await storage.upsert_memory(
        _mem(
            "Prefers thiserror over anyhow for rust error handling",
            Scope.global_(),
            ["rust", "thiserror", "error-handling"],
            category="preference",
        )
    )
    retriever = ScopedRetriever(storage, settings=settings)
    index = await session_start_injection(
        retriever, cwd=str(tmp_path), git_remote="git@github.com:op/rust-cli.git"
    )
    assert index.token_estimate <= settings.inject.token_budget
    assert "thiserror" in index.text


async def test_mid_session_injection_smaller_budget(settings: Settings) -> None:
    storage = InMemoryStorage()
    for i in range(8):
        await storage.upsert_memory(
            _mem(
                f"Prefers approach {i} for async runtime selection in services",
                Scope.global_(),
                ["async", "runtime"],
                category="preference",
            )
        )
    retriever = ScopedRetriever(storage, settings=settings, default_project="svc")
    index = await mid_session_injection(
        retriever,
        "how should we pick the async runtime for this service?",
        project="svc",
        settings=settings,
    )
    # Mid-session draws from a *fraction* of the SessionStart budget (smaller).
    assert index.token_estimate <= settings.inject.token_budget // 2


async def test_mid_session_injection_does_not_record_access(settings: Settings) -> None:
    storage = InMemoryStorage()
    await storage.upsert_memory(
        _mem(
            "Prefers tokio for async runtime",
            Scope.global_(),
            ["async", "runtime", "tokio"],
            category="preference",
        )
    )
    retriever = ScopedRetriever(storage, settings=settings, default_project="svc")
    await mid_session_injection(retriever, "what async runtime do I like?", project="svc")
    # build_index path -> no access recorded (FR-MNT-3).
    assert all(m.access_count == 0 for m in storage.memories.values())
