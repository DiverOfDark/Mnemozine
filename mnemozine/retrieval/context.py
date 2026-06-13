"""Working-context detection for proactive injection (FR-RET-3).

On a Claude Code ``SessionStart`` (and mid-session, FR-RET-5) the retriever needs
a :class:`~mnemozine.interfaces.RetrievalContext` describing *where the operator
is working* so retrieval can be scoped (FR-RET-2). The PRD lists the signals:
cwd, ``Cargo.toml`` / ``package.json``, git remote, and recent turns.

This module derives that context **offline and cheaply** (filesystem reads +
small regexes only; no LLM, no network) so the SessionStart hook stays fast:

* the *project* id from the git remote (preferred — stable across machines) else
  the directory name,
* candidate *entities* from manifest files (crate/package name + dependency
  names) and from any recent-turn text,
* the composed *scopes*: ``project:<id>`` + ``global`` (FR-RET-2 composition).

It depends only on the schema/interfaces (no sibling module), so it is safe for
the retrieval layer to own.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from mnemozine.interfaces import RetrievalContext
from mnemozine.schema.models import Scope

# Conservative cap on how many entities we derive from context so the
# downstream neighborhood stays bounded (FR-RET-2 "roughly constant" subset).
_MAX_CONTEXT_ENTITIES = 12

# A git remote like ``git@github.com:op/rust-cli.git`` or
# ``https://github.com/op/rust-cli`` -> capture the repo name (last path part).
_GIT_REMOTE_RE = re.compile(r"[/:]([^/:]+?)(?:\.git)?$")

# Tokenize recent-turn text into candidate entity-ish words (identifiers,
# hyphenated tech terms). Lowercased, deduped, stopword-filtered downstream.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")

# Tiny stopword set so recent-text entity derivation does not surface filler.
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "you",
        "your",
        "are",
        "was",
        "were",
        "from",
        "have",
        "has",
        "but",
        "not",
        "can",
        "will",
        "would",
        "should",
        "could",
        "about",
        "into",
        "over",
        "what",
        "when",
        "how",
        "why",
        "use",
        "using",
        "used",
        "let",
        "get",
        "set",
        "add",
        "new",
        "old",
        "all",
        "any",
        "some",
        "more",
        "most",
        "now",
        "then",
        "here",
        "there",
    }
)


def _dedup_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def project_from_git_remote(remote: str | None) -> str | None:
    """Derive a stable project id from a git remote URL (FR-RET-3).

    Preferred over the directory name because it is stable across checkouts /
    machines. Returns ``None`` when ``remote`` is empty or unparseable.
    """

    if not remote:
        return None
    match = _GIT_REMOTE_RE.search(remote.strip())
    if not match:
        return None
    name = match.group(1).strip()
    return name or None


def _read_git_remote(cwd: Path) -> str | None:
    """Best-effort read of ``origin`` from ``.git/config`` without shelling out.

    Walks up from ``cwd`` to find a ``.git/config`` and extracts the first
    ``url = ...`` line (which is ``[remote "origin"]`` in the common case). Stays
    offline and dependency-free; returns ``None`` if nothing is found.
    """

    for directory in [cwd, *cwd.parents]:
        config = directory / ".git" / "config"
        if config.is_file():
            try:
                text = config.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
            in_origin = False
            for raw in text.splitlines():
                line = raw.strip()
                if line.startswith("["):
                    in_origin = line == '[remote "origin"]'
                    continue
                if in_origin and line.startswith("url"):
                    _, _, value = line.partition("=")
                    return value.strip() or None
            return None
    return None


def _entities_from_cargo(path: Path) -> list[str]:
    """Crate name + dependency names from a ``Cargo.toml`` (FR-RET-3)."""

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    out: list[str] = []
    pkg = data.get("package")
    if isinstance(pkg, dict):
        name = pkg.get("name")
        if isinstance(name, str):
            out.append(name)
    for key in ("dependencies", "dev-dependencies", "build-dependencies"):
        deps = data.get(key)
        if isinstance(deps, dict):
            out.extend(d for d in deps if isinstance(d, str))
    # Rust manifests imply the language as an entity.
    out.append("rust")
    return out


def _entities_from_package_json(path: Path) -> list[str]:
    """Package name + dependency names from a ``package.json`` (FR-RET-3)."""

    import json

    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    out: list[str] = []
    name = data.get("name")
    if isinstance(name, str):
        # Strip an npm scope (``@scope/pkg`` -> ``pkg``).
        out.append(name.split("/")[-1])
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        deps = data.get(key)
        if isinstance(deps, dict):
            out.extend(d.split("/")[-1] for d in deps if isinstance(d, str))
    out.append("javascript")
    return out


def entities_from_text(text: str | None, *, limit: int = _MAX_CONTEXT_ENTITIES) -> list[str]:
    """Derive candidate entities from recent-turn text (FR-RET-3).

    Lowercased identifier-ish words, stopword-filtered and deduped, capped at
    ``limit``. Purely lexical (no LLM) so it is fast and offline.
    """

    if not text:
        return []
    words = [w.lower() for w in _WORD_RE.findall(text)]
    words = [w for w in words if w not in _STOPWORDS]
    return _dedup_keep_order(words)[:limit]


def detect_context(
    *,
    cwd: str | Path | None = None,
    git_remote: str | None = None,
    recent_text: str | None = None,
    extra_entities: list[str] | None = None,
) -> RetrievalContext:
    """Build a :class:`RetrievalContext` from the working environment (FR-RET-3).

    Resolution order for the project id: explicit/derived git remote first
    (stable), then the cwd directory name. Entities are gathered from
    ``Cargo.toml`` / ``package.json`` manifests under ``cwd``, from
    ``recent_text``, and from ``extra_entities`` (e.g. entities the ingest layer
    already tagged), then deduped and capped.

    ``scopes`` is the FR-RET-2 composition: ``project:<id>`` (when a project is
    known) **plus** ``global`` — never the whole graph. All filesystem reads are
    best-effort; a missing/unreadable file simply contributes nothing.
    """

    cwd_path = Path(cwd).expanduser() if cwd is not None else Path.cwd()

    remote = git_remote if git_remote is not None else _read_git_remote(cwd_path)
    project = project_from_git_remote(remote)
    if project is None:
        # Fall back to the directory name (ignore filesystem roots).
        name = cwd_path.name
        project = name or None

    entities: list[str] = []
    cargo = cwd_path / "Cargo.toml"
    if cargo.is_file():
        entities.extend(_entities_from_cargo(cargo))
    package_json = cwd_path / "package.json"
    if package_json.is_file():
        entities.extend(_entities_from_package_json(package_json))
    entities.extend(entities_from_text(recent_text))
    if extra_entities:
        entities.extend(e.lower() for e in extra_entities)

    entities = _dedup_keep_order([e.lower() for e in entities])[:_MAX_CONTEXT_ENTITIES]

    scopes: list[Scope] = []
    if project:
        scopes.append(Scope.project(project))
    scopes.append(Scope.global_())

    return RetrievalContext(
        project=project,
        scopes=scopes,
        entities=entities,
        recent_text=recent_text,
    )
