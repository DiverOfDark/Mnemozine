"""Tests for the ``mnemozine-maintenance migrate`` subcommand + the startup hook.

Both drive the migration runner over an offline Container (conftest fakes), so no
FalkorDB / Ollama / Qwen is needed:

* the ``migrate`` subcommand runs the pending migrations over the live Container
  storage, prints the report, and upgrades records IN PLACE; ``--dry-run`` reports
  the plan without writing;
* the ``_apply_startup_migrations`` hook warns on a stale store (``warn_on_stale``)
  and auto-applies the cheap baseline (``auto_on_startup``), never blocking startup.
"""

from __future__ import annotations

import logging
from pathlib import Path

from typer.testing import CliRunner

from mnemozine.app import Container, _apply_startup_migrations, maintenance_app
from mnemozine.config import MigrateSettings, Settings
from mnemozine.migrations import CURRENT_DATA_VERSION
from mnemozine.schema.models import MemoryUnit, Provenance, Scope
from tests.conftest import FakeEmbeddingProvider, FakeLLMProvider, InMemoryStorage

runner = CliRunner()

_WORKTREE_RAW_PATH = (
    "/home/op/Projects/rust-cli/.claude/worktrees/agent-7f3a/transcript.jsonl"
)


def _offline_container(storage: InMemoryStorage, **migrate_flags: bool) -> Container:
    """A Container wired to offline fakes, with ``settings.migrate.*`` overrides."""

    settings = Settings()
    settings.web.static_dir = Path("/nonexistent-spa-dir-for-tests")
    if migrate_flags:
        settings.migrate = MigrateSettings(**migrate_flags)
    c = Container(settings=settings)
    c._storage = storage
    c._embedding = FakeEmbeddingProvider()
    c._llm = FakeLLMProvider()
    return c


def _v0_worktree_memory(storage: InMemoryStorage) -> MemoryUnit:
    m = MemoryUnit(
        content="Use thiserror for Rust error handling.",
        scope=Scope.project("agent-7f3a"),  # the worktree-scope bug
        provenance=Provenance(
            source="claude_code", session_id="sess-9", raw_path=_WORKTREE_RAW_PATH
        ),
        data_version=0,
    )
    storage.memories[m.id] = m
    return m


# ---------------------------------------------------------------------------
# The migrate subcommand
# ---------------------------------------------------------------------------


def test_migrate_subcommand_is_registered() -> None:
    names = {c.name for c in maintenance_app.registered_commands}
    assert "migrate" in names


def test_migrate_applies_pending_in_place(monkeypatch) -> None:
    storage = InMemoryStorage()
    memory = _v0_worktree_memory(storage)
    container = _offline_container(storage)
    monkeypatch.setattr(Container, "from_env", classmethod(lambda cls: container))

    result = runner.invoke(maintenance_app, ["migrate"])

    assert result.exit_code == 0, result.output
    assert "data_version 0 -> 1" in result.output
    # Applied in place: same id, scope re-derived, stamped to v1.
    assert memory.id in storage.memories
    assert storage.memories[memory.id].scope.as_str() == "project:rust-cli"
    assert storage.memories[memory.id].data_version == CURRENT_DATA_VERSION


def test_migrate_dry_run_does_not_write(monkeypatch) -> None:
    storage = InMemoryStorage()
    memory = _v0_worktree_memory(storage)
    container = _offline_container(storage)
    monkeypatch.setattr(Container, "from_env", classmethod(lambda cls: container))

    result = runner.invoke(maintenance_app, ["migrate", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    assert "PLAN v1" in result.output
    # Nothing written: still v0, still the stale worktree scope.
    assert storage.memories[memory.id].data_version == 0
    assert storage.memories[memory.id].scope.as_str() == "project:agent-7f3a"


def test_migrate_is_no_op_when_current(monkeypatch) -> None:
    storage = InMemoryStorage()  # empty store -> already at CURRENT
    container = _offline_container(storage)
    monkeypatch.setattr(Container, "from_env", classmethod(lambda cls: container))

    result = runner.invoke(maintenance_app, ["migrate"])
    assert result.exit_code == 0, result.output
    assert "applied=0" in result.output


# ---------------------------------------------------------------------------
# The startup hook
# ---------------------------------------------------------------------------


async def test_startup_warns_on_stale_store(caplog) -> None:
    storage = InMemoryStorage()
    memory = _v0_worktree_memory(storage)
    container = _offline_container(
        storage, auto_on_startup=False, warn_on_stale=True
    )

    with caplog.at_level(logging.WARNING, logger="mnemozine.app"):
        await _apply_startup_migrations(container)

    assert any("run: mnemozine-maintenance migrate" in r.message for r in caplog.records)
    # Warn-only: nothing migrated.
    assert storage.memories[memory.id].data_version == 0


async def test_startup_auto_applies_cheap_baseline() -> None:
    storage = InMemoryStorage()
    memory = _v0_worktree_memory(storage)
    container = _offline_container(
        storage, auto_on_startup=True, warn_on_stale=True
    )

    await _apply_startup_migrations(container)

    # Auto-applied in place: scope fixed + stamped to v1.
    assert storage.memories[memory.id].scope.as_str() == "project:rust-cli"
    assert storage.memories[memory.id].data_version == CURRENT_DATA_VERSION
    assert await storage.min_data_version() == CURRENT_DATA_VERSION


async def test_startup_hook_never_raises_on_failure(monkeypatch, caplog) -> None:
    """A migration problem is caught + logged, never blocks startup."""

    storage = InMemoryStorage()
    _v0_worktree_memory(storage)
    container = _offline_container(storage, auto_on_startup=True)

    async def _boom() -> int:
        raise RuntimeError("min_data_version blew up")

    monkeypatch.setattr(storage, "min_data_version", _boom)

    with caplog.at_level(logging.ERROR, logger="mnemozine.app"):
        # Must NOT raise.
        await _apply_startup_migrations(container)

    assert any("startup migration check failed" in r.message for r in caplog.records)


async def test_startup_noop_when_both_flags_off() -> None:
    storage = InMemoryStorage()
    memory = _v0_worktree_memory(storage)
    container = _offline_container(
        storage, auto_on_startup=False, warn_on_stale=False
    )
    await _apply_startup_migrations(container)
    # Disabled entirely: untouched.
    assert storage.memories[memory.id].data_version == 0
