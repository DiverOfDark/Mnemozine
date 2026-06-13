"""Unit tests for the Claude Code IngestSource (FR-ING-2/5/6/7).

Covers config-dir resolution (CLAUDE_CONFIG_DIR override), transcript discovery,
backfill event production, content-hash de-dup across a backfill re-run, and the
SourceSession provenance record — all offline against the fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemozine.config import Settings
from mnemozine.ingestion.claude_code.source import (
    CLAUDE_CONFIG_DIR_ENV,
    ClaudeCodeSource,
    resolve_config_dir,
)
from mnemozine.interfaces import IngestSource
from mnemozine.schema.events import Source

FIXTURES = Path(__file__).parent / "fixtures"


def _settings_for_fixtures(tmp_root: Path) -> Settings:
    # Point claude_config_dir at a temp tree whose `projects/` is the fixtures dir.
    return Settings(ingest={"claude_config_dir": str(tmp_root)})


def test_source_is_ingest_source() -> None:
    assert isinstance(ClaudeCodeSource(Settings()), IngestSource)
    assert ClaudeCodeSource().source_name == Source.CLAUDE_CODE.value


def test_resolve_config_dir_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CLAUDE_CONFIG_DIR_ENV, "/custom/claude")
    s = Settings()
    assert resolve_config_dir(s) == Path("/custom/claude")


def test_resolve_config_dir_setting_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CLAUDE_CONFIG_DIR_ENV, raising=False)
    s = Settings(ingest={"claude_config_dir": "/cfg/here"})
    assert resolve_config_dir(s) == Path("/cfg/here")


def test_discover_transcripts(tmp_path: Path) -> None:
    # Build a config tree: <cfg>/projects/<encoded>/<session>.jsonl
    projects = tmp_path / "projects" / "-home-op-Projects-rust-cli"
    projects.mkdir(parents=True)
    src_file = FIXTURES / "-home-op-Projects-rust-cli" / "sess-rust-1.jsonl"
    (projects / "sess-rust-1.jsonl").write_text(src_file.read_text())
    s = _settings_for_fixtures(tmp_path)
    source = ClaudeCodeSource(s)
    found = source.discover_transcripts()
    assert len(found) == 1
    assert found[0].name == "sess-rust-1.jsonl"


@pytest.mark.asyncio
async def test_backfill_yields_events(tmp_path: Path) -> None:
    projects = tmp_path / "projects" / "-home-op-Projects-rust-cli"
    projects.mkdir(parents=True)
    src_file = FIXTURES / "-home-op-Projects-rust-cli" / "sess-rust-1.jsonl"
    (projects / "sess-rust-1.jsonl").write_text(src_file.read_text())
    source = ClaudeCodeSource(_settings_for_fixtures(tmp_path))

    events = [e async for e in source.backfill()]
    assert len(events) == 5
    assert all(e.source is Source.CLAUDE_CODE for e in events)
    assert all(e.tool_calls is None for e in events)  # FR-ING-7 default strip
    assert events[0].project == "rust-cli"


@pytest.mark.asyncio
async def test_backfill_dedups_on_rerun(tmp_path: Path) -> None:
    # Re-running backfill on the same source instance yields nothing the second
    # time (content-hash de-dup, FR-ING-5).
    projects = tmp_path / "projects" / "-home-op-Projects-rust-cli"
    projects.mkdir(parents=True)
    src_file = FIXTURES / "-home-op-Projects-rust-cli" / "sess-rust-1.jsonl"
    (projects / "sess-rust-1.jsonl").write_text(src_file.read_text())
    source = ClaudeCodeSource(_settings_for_fixtures(tmp_path))

    first = [e async for e in source.backfill()]
    second = [e async for e in source.backfill()]
    assert len(first) == 5
    assert second == []


@pytest.mark.asyncio
async def test_backfill_multiple_sessions(tmp_path: Path) -> None:
    proj = tmp_path / "projects" / "-home-op-Projects-rust-cli"
    proj.mkdir(parents=True)
    for name in ("sess-rust-1.jsonl", "sess-rewind.jsonl"):
        (proj / name).write_text(
            (FIXTURES / "-home-op-Projects-rust-cli" / name).read_text()
        )
    source = ClaudeCodeSource(_settings_for_fixtures(tmp_path))
    events = [e async for e in source.backfill()]
    sessions = {e.session_id for e in events}
    assert sessions == {"sess-rust-1", "sess-rewind"}


def test_session_for_builds_provenance(tmp_path: Path) -> None:
    proj = tmp_path / "projects" / "-home-op-Projects-rust-cli"
    proj.mkdir(parents=True)
    transcript = proj / "sess-rust-1.jsonl"
    transcript.write_text(
        (FIXTURES / "-home-op-Projects-rust-cli" / "sess-rust-1.jsonl").read_text()
    )
    source = ClaudeCodeSource(_settings_for_fixtures(tmp_path))
    session = source.session_for(transcript)
    assert session.source == Source.CLAUDE_CODE.value
    assert session.session_id == "sess-rust-1"
    assert session.project == "rust-cli"
    assert session.raw_path == str(transcript)
    assert session.started_at is not None and session.ended_at is not None
    assert session.started_at <= session.ended_at
