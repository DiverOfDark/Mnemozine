"""Unit tests for the BASELINE migration to data_version 1 (the worktree fix).

The baseline migration is the framework's seed + demonstration: a CHEAP, no-LLM
RECLASSIFY that re-derives scope from STORED provenance+content and stamps every
below-v1 record to v1, IN PLACE (same ids, no deletes). These tests drive it over
the conftest ``InMemoryStorage`` fake:

* the worktree-scope bug is fixed in place — a v0 memory wrongly scoped to an
  opaque ``project:<worktree-id>`` is re-derived to its real ``project:<name>``,
  stamped to v1, and keeps the SAME id (no new node, no delete);
* idempotency (FR-MNT-5) — a re-run finds nothing below v1 and changes nothing;
* the chunk tier is advanced too, so ``min_data_version`` actually reaches v1;
* the migration is registered in the real ``MIGRATIONS`` registry at version 1.
"""

from __future__ import annotations

import mnemozine.migrations as migrations_pkg
from mnemozine.interfaces import MaintenanceReport
from mnemozine.migrations import CURRENT_DATA_VERSION, Migration
from mnemozine.migrations.baseline import (
    BASELINE_MIGRATION,
    BaselineReclassifyMigration,
    rederive_scope,
)
from mnemozine.schema.models import MemoryUnit, Provenance, RawChunk, Scope
from tests.conftest import InMemoryStorage

# A literal worktree transcript path (subagent ran in a git worktree). The FIXED
# scope derivation rolls this up to the real project, not the opaque worktree id.
_WORKTREE_RAW_PATH = (
    "/home/op/Projects/rust-cli/.claude/worktrees/agent-7f3a/transcript.jsonl"
)
_REAL_PROJECT = "rust-cli"


def _v0_worktree_memory(storage: InMemoryStorage) -> MemoryUnit:
    """Seed a v0 memory wrongly scoped to the opaque worktree id."""

    m = MemoryUnit(
        content="Use thiserror over anyhow for Rust error handling.",
        # The BUG: scoped to the opaque worktree leaf instead of the project.
        scope=Scope.project("agent-7f3a"),
        category="preference",
        entities=["rust", "thiserror"],
        provenance=Provenance(
            source="claude_code",
            session_id="sess-9",
            raw_path=_WORKTREE_RAW_PATH,
        ),
        data_version=0,
    )
    storage.memories[m.id] = m
    return m


# ---------------------------------------------------------------------------
# Registry + protocol
# ---------------------------------------------------------------------------


def test_baseline_satisfies_migration_protocol() -> None:
    assert isinstance(BASELINE_MIGRATION, Migration)


def test_baseline_is_registered_at_version_1() -> None:
    versions = [m.version for m in migrations_pkg.MIGRATIONS]
    assert 1 in versions
    assert BASELINE_MIGRATION.version == 1
    assert BASELINE_MIGRATION.version == CURRENT_DATA_VERSION
    # Cheap reclassify -> safe to auto-run at startup.
    assert BASELINE_MIGRATION.requires_reextract is False


def test_register_is_idempotent() -> None:
    from mnemozine.migrations.baseline import register

    before = [m.version for m in migrations_pkg.MIGRATIONS]
    register()
    register()
    after = [m.version for m in migrations_pkg.MIGRATIONS]
    # No duplicate baseline entries from repeated registration.
    assert after.count(1) == before.count(1) == 1


# ---------------------------------------------------------------------------
# rederive_scope (the cheap, no-LLM derivation)
# ---------------------------------------------------------------------------


def test_rederive_scope_rolls_worktree_up_to_project() -> None:
    m = MemoryUnit(
        content="x",
        scope=Scope.project("agent-7f3a"),
        provenance=Provenance(
            source="claude_code", session_id="s", raw_path=_WORKTREE_RAW_PATH
        ),
        data_version=0,
    )
    new = rederive_scope(m)
    assert new is not None
    assert new.as_str() == f"project:{_REAL_PROJECT}"


def test_rederive_scope_none_without_raw_path() -> None:
    m = MemoryUnit(
        content="x",
        scope=Scope.global_(),
        provenance=Provenance(source="claude_code", session_id="s"),
        data_version=0,
    )
    assert rederive_scope(m) is None


def test_rederive_scope_none_for_classify_sentinel() -> None:
    # Default provenance is the classify sentinel (no real session/transcript).
    m = MemoryUnit(content="x", scope=Scope.global_(), data_version=0)
    assert rederive_scope(m) is None


# ---------------------------------------------------------------------------
# The migration run() — in place, idempotent, both tiers
# ---------------------------------------------------------------------------


async def test_run_rescopes_in_place_same_id_no_delete() -> None:
    storage = InMemoryStorage()
    memory = _v0_worktree_memory(storage)
    original_id = memory.id
    ids_before = set(storage.memories)

    report = await BaselineReclassifyMigration().run(storage)

    assert isinstance(report, MaintenanceReport)
    # SAME ids: in-place upgrade, never a wipe + re-insert.
    assert set(storage.memories) == ids_before
    assert original_id in storage.memories
    migrated = storage.memories[original_id]
    # Scope re-derived to the real project (worktree-scope bug fixed in place).
    assert migrated.scope.as_str() == f"project:{_REAL_PROJECT}"
    # Stamped to v1.
    assert migrated.data_version == CURRENT_DATA_VERSION
    # Content/entities untouched (a re-tag, not a re-extract).
    assert migrated.content == memory.content
    assert migrated.entities == ["rust", "thiserror"]
    # min_data_version now reaches the target.
    assert await storage.min_data_version() == CURRENT_DATA_VERSION


async def test_run_is_idempotent_no_op_on_rerun() -> None:
    storage = InMemoryStorage()
    memory = _v0_worktree_memory(storage)

    first = await BaselineReclassifyMigration().run(storage)
    assert first.re_extracted >= 1
    scope_after_first = storage.memories[memory.id].scope.as_str()

    second = await BaselineReclassifyMigration().run(storage)
    # Nothing below v1 anymore: no records touched.
    assert second.re_extracted == 0
    # Scope unchanged on the no-op re-run.
    assert storage.memories[memory.id].scope.as_str() == scope_after_first


async def test_run_stamps_already_correct_memory_in_place() -> None:
    """A correctly-scoped v0 memory is stamped to v1 without changing its scope."""

    storage = InMemoryStorage()
    # raw_path under ~/.claude/projects/<encoded>/ derives back to the SAME scope.
    raw = "/home/op/.claude/projects/-home-op-Projects-rust-cli/sess-1.jsonl"
    correct_scope = Scope.project("cli")  # what the path-only derivation yields
    m = MemoryUnit(
        content="pins tokio 1.38",
        scope=correct_scope,
        provenance=Provenance(
            source="claude_code", session_id="s1", raw_path=raw
        ),
        data_version=0,
    )
    storage.memories[m.id] = m

    await BaselineReclassifyMigration().run(storage)

    after = storage.memories[m.id]
    # Scope unchanged (re-derivation matches) but version stamped up in place.
    assert after.scope.as_str() == correct_scope.as_str()
    assert after.data_version == CURRENT_DATA_VERSION


async def test_run_advances_raw_chunk_tier() -> None:
    """The chunk tier must also reach v1, or min_data_version never does."""

    storage = InMemoryStorage()
    _v0_worktree_memory(storage)
    chunk = RawChunk(
        content_hash="abc123",
        content="raw chunk text",
        source="claude_code",
        session_id="sess-9",
        scope=Scope.project("rust-cli"),
        project="rust-cli",
        data_version=0,
    )
    storage.raw_chunks[chunk.content_hash] = chunk

    await BaselineReclassifyMigration().run(storage)

    # Both tiers advanced: the stale chunk is stamped without re-extracting.
    assert storage.raw_chunks["abc123"].data_version == CURRENT_DATA_VERSION
    assert await storage.min_data_version() == CURRENT_DATA_VERSION


async def test_run_extractor_arg_is_ignored() -> None:
    """The cheap path ignores any extractor passed in (no GPU/LLM)."""

    storage = InMemoryStorage()
    _v0_worktree_memory(storage)

    class _ExplodingExtractor:
        async def extract(self, chunk):  # type: ignore[no-untyped-def]
            raise AssertionError("extractor must not be used by the cheap path")

        async def classify(self, statement, context):  # type: ignore[no-untyped-def]
            raise AssertionError("classify must not be used by the cheap path")

    report = await BaselineReclassifyMigration().run(
        storage, extractor=_ExplodingExtractor()
    )
    assert report.re_extracted >= 1


async def test_run_no_op_on_empty_store() -> None:
    storage = InMemoryStorage()
    report = await BaselineReclassifyMigration().run(storage)
    assert report.re_extracted == 0
