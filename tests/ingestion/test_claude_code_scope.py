"""Unit tests for Claude Code transcript -> hierarchical scope derivation (FR-EXT-3).

``derive_scope_from_transcript`` maps a Claude Code transcript path to a
hierarchical :class:`~mnemozine.schema.models.Scope`. These tests pin the two
load-bearing behaviors of the core data-model redesign:

* the parent PROJECT is the top-level ``$CLAUDE_CONFIG_DIR/projects/<encoded-cwd>``
  dir, DECODED to a friendly name (the last path component of the encoded cwd),
  with the literal ``cwd`` leaf winning when supplied;
* SUBAGENT / WORKFLOW transcripts living under ``.../<encoded-cwd>/<session>/
  subagents/...`` ROLL UP to that same project — they NEVER get an opaque
  ``project:agent-XXXX`` scope (the bug this redesign fixes). With
  ``ScopeSettings.subagent_subsegments`` on, the workflow id is appended as a
  sub-segment so it composes (and never leaks across) the project.

All paths are synthetic strings: derivation is pure path manipulation (no disk
reads), so no transcript file needs to exist on disk.
"""

from __future__ import annotations

import pytest

from mnemozine.config import ScopeSettings, Settings
from mnemozine.ingestion.claude_code.parser import (
    decode_project_dirname,
    derive_scope_from_transcript,
)
from mnemozine.schema.models import Scope

# A realistic encoded project dir for /var/home/diverofdark/Projects/Mnemozine.
ENCODED = "-var-home-diverofdark-Projects-Mnemozine"
PROJECT = "Mnemozine"

# A top-level session transcript: .../projects/<encoded>/<session>.jsonl
TOP_LEVEL = f"/home/u/.claude/projects/{ENCODED}/3f6bdbf0-9fbd.jsonl"

# A DEEP subagent/workflow transcript:
#   .../projects/<encoded>/<session>/subagents/workflows/wf_<id>/agent-<id>.jsonl
DEEP_SUBAGENT = (
    f"/home/u/.claude/projects/{ENCODED}/3f6bdbf0-9fbd/"
    "subagents/workflows/wf_abc123/agent-DEADBEEF.jsonl"
)


# ---------------------------------------------------------------------------
# decode_project_dirname — the encoded-cwd -> friendly-name decode
# ---------------------------------------------------------------------------


def test_decode_project_dirname_takes_decoded_leaf() -> None:
    assert decode_project_dirname(ENCODED) == PROJECT
    assert decode_project_dirname("-home-op-Projects-rust-cli") == "cli"
    # Tolerant of leading/trailing separators and empty input.
    assert decode_project_dirname("-Mnemozine-") == "Mnemozine"
    assert decode_project_dirname("") == ""


# ---------------------------------------------------------------------------
# Project derivation: top-level session -> project:<decoded-name>
# ---------------------------------------------------------------------------


def test_top_level_session_scopes_to_decoded_project() -> None:
    scope = derive_scope_from_transcript(TOP_LEVEL, Settings())
    assert scope == Scope.project(PROJECT)
    assert scope.as_str() == "project:Mnemozine"
    assert scope.project_id == "Mnemozine"


def test_literal_cwd_leaf_wins_over_encoded_dir() -> None:
    # The literal cwd is the real working dir, not the lossy encoded form: it wins.
    scope = derive_scope_from_transcript(
        TOP_LEVEL, Settings(), cwd="/var/home/diverofdark/Projects/Mnemozine"
    )
    assert scope == Scope.project("Mnemozine")


# ---------------------------------------------------------------------------
# Subagent / workflow ROLL-UP (the project:agent-XXXX bug fix)
# ---------------------------------------------------------------------------


def test_deep_subagent_transcript_rolls_up_to_parent_project() -> None:
    """A deep subagent transcript must NOT get an opaque project:agent-XXXX scope."""

    scope = derive_scope_from_transcript(DEEP_SUBAGENT, Settings())
    # Rolls up to the SAME parent project as the top-level session (default off:
    # collapse to the bare project, never a sibling/opaque agent scope).
    assert scope == Scope.project(PROJECT)
    assert scope.as_str() == "project:Mnemozine"
    # Explicitly: it is NOT scoped to the agent id segment.
    assert "agent-DEADBEEF" not in scope.as_str()
    assert scope.leaf == "Mnemozine"


def test_top_level_and_subagent_share_the_same_project_scope() -> None:
    top = derive_scope_from_transcript(TOP_LEVEL, Settings())
    sub = derive_scope_from_transcript(DEEP_SUBAGENT, Settings())
    # Roll-up guarantee: a subagent run composes with its parent session's project.
    assert top == sub


def test_subagent_subsegments_append_workflow_id() -> None:
    """With sub-segmenting on, the workflow id rolls up AS a sub-segment."""

    settings = Settings(scope=ScopeSettings(subagent_subsegments=True))
    scope = derive_scope_from_transcript(DEEP_SUBAGENT, settings)
    assert scope == Scope.project(PROJECT, "wf_abc123")
    assert scope.as_str() == "project:Mnemozine/wf_abc123"
    # The workflow sub-scope is still UNDER the project (no-leak): the project is
    # an ancestor, so the subagent memory composes with project + global, never a
    # sibling project.
    assert Scope.project(PROJECT).contains(scope)
    assert scope.is_descendant_of(Scope.project(PROJECT))
    # ancestors compose root-first, self-last.
    assert [s.as_str() for s in scope.ancestors()] == [
        "global",
        "project:Mnemozine",
        "project:Mnemozine/wf_abc123",
    ]


def test_subagent_subsegments_off_collapses_to_project() -> None:
    # Default off: the same deep path collapses to the bare project scope.
    settings = Settings(scope=ScopeSettings(subagent_subsegments=False))
    scope = derive_scope_from_transcript(DEEP_SUBAGENT, settings)
    assert scope == Scope.project(PROJECT)


def test_subagent_without_workflow_segment_rolls_up_to_project() -> None:
    # A subagent transcript with no wf_<id> segment still rolls up to the project
    # even with sub-segmenting on (there is no sub-segment to append).
    path = (
        f"/home/u/.claude/projects/{ENCODED}/sess-1/subagents/agent-XYZ.jsonl"
    )
    settings = Settings(scope=ScopeSettings(subagent_subsegments=True))
    scope = derive_scope_from_transcript(path, settings)
    assert scope == Scope.project(PROJECT)


# ---------------------------------------------------------------------------
# Robustness: no projects/ ancestor on the path
# ---------------------------------------------------------------------------


def test_no_projects_ancestor_falls_back_to_parent_dir() -> None:
    # A path with no projects/ ancestor: fall back to the parent dir name decode.
    scope = derive_scope_from_transcript("/tmp/-x-y-rust-cli/sess.jsonl", Settings())
    assert scope == Scope.project("cli")


@pytest.mark.parametrize("subseg", [True, False])
def test_derivation_never_yields_global_for_a_project_transcript(subseg: bool) -> None:
    settings = Settings(scope=ScopeSettings(subagent_subsegments=subseg))
    for path in (TOP_LEVEL, DEEP_SUBAGENT):
        scope = derive_scope_from_transcript(path, settings)
        assert not scope.is_global
        assert scope.project_id == PROJECT
