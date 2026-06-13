"""FR-RET-3 working-context detection tests (offline, filesystem-based)."""

from __future__ import annotations

import json
from pathlib import Path

from mnemozine.retrieval.context import (
    detect_context,
    entities_from_text,
    project_from_git_remote,
)


def test_project_from_git_remote_ssh_and_https() -> None:
    assert project_from_git_remote("git@github.com:op/rust-cli.git") == "rust-cli"
    assert project_from_git_remote("https://github.com/op/rust-cli") == "rust-cli"
    assert project_from_git_remote("https://github.com/op/rust-cli.git") == "rust-cli"
    assert project_from_git_remote(None) is None
    assert project_from_git_remote("") is None


def test_entities_from_text_filters_stopwords() -> None:
    ents = entities_from_text("I prefer thiserror over anyhow for error-handling in rust")
    assert "thiserror" in ents
    assert "anyhow" in ents
    assert "error-handling" in ents
    assert "rust" in ents
    # Stopwords dropped.
    assert "the" not in ents
    assert "for" not in ents


def test_detect_context_cargo_manifest(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text(
        """
[package]
name = "rust-cli"

[dependencies]
tokio = "1.38"
thiserror = "1"
""",
        encoding="utf-8",
    )
    ctx = detect_context(cwd=tmp_path, git_remote="git@github.com:op/rust-cli.git")
    assert ctx.project == "rust-cli"
    # Manifest-derived entities present + the rust language entity.
    assert "tokio" in ctx.entities
    assert "thiserror" in ctx.entities
    assert "rust" in ctx.entities
    # Scopes composed: project + global (FR-RET-2).
    scope_strs = {s.as_str() for s in ctx.scopes}
    assert "project:rust-cli" in scope_strs
    assert "global" in scope_strs


def test_detect_context_package_json(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        json.dumps(
            {
                "name": "@scope/web-app",
                "dependencies": {"react": "^18", "@types/node": "^20"},
            }
        ),
        encoding="utf-8",
    )
    ctx = detect_context(cwd=tmp_path)
    # npm scope stripped from the package name.
    assert ctx.project == tmp_path.name  # no git remote -> dir name
    assert "web-app" in ctx.entities
    assert "react" in ctx.entities
    assert "javascript" in ctx.entities


def test_detect_context_falls_back_to_dir_name(tmp_path: Path) -> None:
    ctx = detect_context(cwd=tmp_path)
    assert ctx.project == tmp_path.name


def test_detect_context_reads_git_config(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text(
        '[remote "origin"]\n\turl = git@github.com:op/cool-proj.git\n',
        encoding="utf-8",
    )
    ctx = detect_context(cwd=tmp_path)
    assert ctx.project == "cool-proj"


def test_detect_context_merges_recent_text_entities(tmp_path: Path) -> None:
    ctx = detect_context(
        cwd=tmp_path,
        git_remote="git@github.com:op/proj.git",
        recent_text="let's discuss the async-runtime and cli-parsing design",
    )
    assert "async-runtime" in ctx.entities
    assert "cli-parsing" in ctx.entities
