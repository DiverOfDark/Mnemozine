"""ProvenanceRescopeJob tests — deterministic provenance re-scope (no LLM).

Covers the offline pass that repairs mis-globalized project memos by parsing the
source project from each active global memo's ``provenance.raw_path`` and moving
it global -> project:<its own source project> via the scope-only
:meth:`StorageBackend.reclassify_memory`. Pure/deterministic — no LLM, no
embeddings, no raw transcript text — so it runs offline against the conftest
:class:`~tests.conftest.InMemoryStorage` fake (which already implements
``iter_memories(scope=...)`` + scope-only ``reclassify_memory``; no fake change
needed).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mnemozine.app import Container, maintenance_app
from mnemozine.config import Settings
from mnemozine.interfaces import MaintenanceJob
from mnemozine.maintenance import ProvenanceRescopeJob as ProvenanceRescopeJobExport
from mnemozine.maintenance.provenance_rescope import ProvenanceRescopeJob
from mnemozine.maintenance.runner import build_default_jobs
from mnemozine.schema.models import MemoryUnit, Provenance, Scope
from tests.conftest import (
    FakeEmbeddingProvider,
    FakeLLMProvider,
    InMemoryStorage,
)

_cli = CliRunner()

# Realistic Claude Code transcript paths (the deterministic parse target).
_AIPACK_TOP = (
    "/claude/projects/-var-home-diverofdark-Projects-aipack/sess-1.jsonl"
)
_AIPACK_SUBAGENT = (
    "/claude/projects/-var-home-diverofdark-Projects-aipack/"
    "sess-1/subagents/agent-7f3a.jsonl"
)
_APPBAHN2_TOP = (
    "/claude/projects/-var-home-diverofdark-Projects-AppBahn2/sess-9.jsonl"
)


def _mem(
    content: str,
    *,
    scope: Scope | None = None,
    category: str = "fact",
    raw_path: str | None = _AIPACK_TOP,
    source: str = "claude_code",
    session_id: str = "sess-1",
) -> MemoryUnit:
    return MemoryUnit(
        content=content,
        scope=scope or Scope.global_(),
        category=category,
        confidence=0.9,
        provenance=Provenance(
            source=source, session_id=session_id, raw_path=raw_path
        ),
    )


def test_rescope_satisfies_protocol() -> None:
    job = ProvenanceRescopeJob(InMemoryStorage())
    assert isinstance(job, MaintenanceJob)
    assert job.name == "rescope_global"


@pytest.mark.asyncio
async def test_rescope_moves_global_fact_to_its_source_project() -> None:
    storage = InMemoryStorage()
    mem = _mem("The build pins tokio 1.38.", category="fact", raw_path=_AIPACK_TOP)
    await storage.upsert_memory(mem)

    report = await ProvenanceRescopeJob(storage, settings=Settings()).run()

    assert report.consolidated == 1
    assert storage.memories[mem.id].scope.as_str() == "project:aipack"


@pytest.mark.asyncio
async def test_rescope_rolls_subagent_path_up_to_parent_project() -> None:
    # A subagent/worktree transcript lives UNDER the session dir and must roll up
    # to the bare parent project (never project:agent-7f3a).
    storage = InMemoryStorage()
    mem = _mem(
        "The MCP server exposes recall(query, scope?).",
        category="decision",
        raw_path=_AIPACK_SUBAGENT,
    )
    await storage.upsert_memory(mem)

    await ProvenanceRescopeJob(storage).run()

    assert storage.memories[mem.id].scope.as_str() == "project:aipack"


@pytest.mark.asyncio
async def test_rescope_moves_appbahn2_fact_to_project_appbahn2() -> None:
    storage = InMemoryStorage()
    mem = _mem("A BASELINE migration to version 1.", raw_path=_APPBAHN2_TOP)
    await storage.upsert_memory(mem)

    await ProvenanceRescopeJob(storage).run()

    assert storage.memories[mem.id].scope.as_str() == "project:AppBahn2"


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["preference", "convention", "rule", "idea"])
async def test_rescope_keeps_cross_project_categories_global(category: str) -> None:
    storage = InMemoryStorage()
    mem = _mem("Prefers thiserror over anyhow.", category=category)
    await storage.upsert_memory(mem)

    report = await ProvenanceRescopeJob(storage, settings=Settings()).run()

    assert report.consolidated == 0
    assert storage.memories[mem.id].scope.is_global


@pytest.mark.asyncio
async def test_rescope_skips_missing_raw_path() -> None:
    storage = InMemoryStorage()
    mem = _mem("Some global fact.", raw_path=None)
    await storage.upsert_memory(mem)

    report = await ProvenanceRescopeJob(storage).run()

    assert report.consolidated == 0
    assert storage.memories[mem.id].scope.is_global


@pytest.mark.asyncio
async def test_rescope_skips_ambiguous_path_without_projects_ancestor() -> None:
    # No `projects/` ancestor -> ambiguous, leave at global (do not guess).
    storage = InMemoryStorage()
    mem = _mem("Ambiguous origin.", raw_path="/tmp/random/scratch.jsonl")
    await storage.upsert_memory(mem)

    report = await ProvenanceRescopeJob(storage).run()

    assert report.consolidated == 0
    assert storage.memories[mem.id].scope.is_global


@pytest.mark.asyncio
async def test_rescope_skips_classify_sentinel() -> None:
    # A classify-sentinel provenance has no real originating session.
    storage = InMemoryStorage()
    mem = MemoryUnit(
        content="Sentinel-sourced fact.",
        scope=Scope.global_(),
        category="fact",
        provenance=Provenance.classify_sentinel(),
    )
    await storage.upsert_memory(mem)

    report = await ProvenanceRescopeJob(storage).run()

    assert report.consolidated == 0
    assert storage.memories[mem.id].scope.is_global


@pytest.mark.asyncio
async def test_rescope_never_leaks_to_unrelated_project() -> None:
    # The memo from an aipack transcript must move ONLY to project:aipack — never
    # to an unrelated project, even if other projects exist in the store.
    storage = InMemoryStorage()
    aipack = _mem("aipack fact", raw_path=_AIPACK_TOP)
    appbahn = _mem("appbahn fact", raw_path=_APPBAHN2_TOP)
    await storage.upsert_memory(aipack)
    await storage.upsert_memory(appbahn)

    await ProvenanceRescopeJob(storage).run()

    assert storage.memories[aipack.id].scope.as_str() == "project:aipack"
    assert storage.memories[appbahn.id].scope.as_str() == "project:AppBahn2"


@pytest.mark.asyncio
async def test_rescope_is_idempotent_rerun_is_noop() -> None:
    storage = InMemoryStorage()
    mem = _mem("The build pins tokio 1.38.", raw_path=_AIPACK_TOP)
    await storage.upsert_memory(mem)

    first = await ProvenanceRescopeJob(storage).run()
    second = await ProvenanceRescopeJob(storage).run()

    assert first.consolidated == 1
    # The moved memo left the global iteration -> a re-run touches nothing.
    assert second.consolidated == 0
    assert storage.memories[mem.id].scope.as_str() == "project:aipack"


@pytest.mark.asyncio
async def test_rescope_report_surfaces_counts_and_sample() -> None:
    storage = InMemoryStorage()
    moved = _mem("moved fact", raw_path=_AIPACK_TOP)
    kept = _mem("Prefers thiserror.", category="preference")
    skipped = _mem("no path", raw_path=None)
    await storage.upsert_memory(moved)
    await storage.upsert_memory(kept)
    await storage.upsert_memory(skipped)

    report = await ProvenanceRescopeJob(storage, settings=Settings()).run()

    assert report.consolidated == 1
    blob = "\n".join(report.notes)
    assert "re-scoped 1/3" in blob
    assert "kept 1 cross-project" in blob
    assert "skipped 1 unparseable" in blob
    assert "project:aipack" in blob


# ---------------------------------------------------------------------------
# Wiring: exports, default-job-set exclusion, CLI subcommand
# ---------------------------------------------------------------------------


def test_job_is_exported_from_maintenance_package() -> None:
    # Exported alongside ReclassifyJob from mnemozine.maintenance.
    assert ProvenanceRescopeJobExport is ProvenanceRescopeJob


def test_rescope_is_not_in_default_scheduled_jobs() -> None:
    # Operator-triggered like reclassify/re-extract — must NOT run on every cron
    # tick (a deterministic but store-wide re-scope is an explicit migration).
    jobs = build_default_jobs(
        InMemoryStorage(),
        FakeLLMProvider(),
        FakeEmbeddingProvider(),
        settings=Settings(),
    )
    assert not any(isinstance(j, ProvenanceRescopeJob) for j in jobs)


def test_rescope_global_subcommand_is_registered() -> None:
    names = {c.name for c in maintenance_app.registered_commands}
    assert "rescope-global" in names


def _offline_container(storage: InMemoryStorage) -> Container:
    settings = Settings()
    settings.web.static_dir = Path("/nonexistent-spa-dir-for-tests")
    c = Container(settings=settings)
    c._storage = storage
    c._embedding = FakeEmbeddingProvider()
    c._llm = FakeLLMProvider()
    return c


def test_cli_rescope_global_applies_in_place(monkeypatch) -> None:
    storage = InMemoryStorage()
    mem = _mem("The build pins tokio 1.38.", raw_path=_AIPACK_TOP)
    storage.memories[mem.id] = mem
    container = _offline_container(storage)
    monkeypatch.setattr(Container, "from_env", classmethod(lambda cls: container))

    result = _cli.invoke(maintenance_app, ["rescope-global"])

    assert result.exit_code == 0, result.output
    assert storage.memories[mem.id].scope.as_str() == "project:aipack"


def test_cli_rescope_global_dry_run_does_not_write(monkeypatch) -> None:
    storage = InMemoryStorage()
    mem = _mem("The build pins tokio 1.38.", raw_path=_AIPACK_TOP)
    storage.memories[mem.id] = mem
    container = _offline_container(storage)
    monkeypatch.setattr(Container, "from_env", classmethod(lambda cls: container))

    result = _cli.invoke(maintenance_app, ["rescope-global", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    assert "would re-scope" in result.output
    # Nothing written: still global.
    assert storage.memories[mem.id].scope.is_global
