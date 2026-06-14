"""Parse Claude Code JSONL transcripts into normalized events (FR-ING-2/7).

Claude Code writes one JSON object per line to
``$CLAUDE_CONFIG_DIR/projects/<project>/<session-id>.jsonl``. Lines come in many
``type``s — ``user``, ``assistant``, plus bookkeeping records
(``file-history-snapshot``, ``permission-mode``, ``ai-title``, ``attachment``,
``last-prompt``, ...). Only the conversational ``user`` / ``assistant`` turns
carry durable memory value, so the parser:

1. keeps only conversational turns and drops bookkeeping lines (returns ``None``);
2. derives the ``project`` field from the transcript path / the record's ``cwd``
   (FR-ING-2);
3. flattens the Anthropic message ``content`` (a string or a list of typed
   blocks) into plain text, dropping ``thinking`` blocks and — per FR-ING-7 —
   ``tool_use`` / ``tool_result`` blocks (the highest-density source of raw
   credentials, file dumps, and command output);
4. returns ``None`` for a turn whose only content was tool traffic, so a chunk is
   not padded with empty events.

The parser is intentionally tolerant: a malformed or unrecognized line yields
``None`` rather than raising, so a single bad line never stalls the watcher.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mnemozine.schema.events import IngestEvent, Role, Source
from mnemozine.schema.models import Scope

if TYPE_CHECKING:
    from mnemozine.config import Settings

# Claude Code transcript line ``type`` values that carry a conversational turn.
_CONVERSATIONAL_TYPES = frozenset({"user", "assistant"})

# Anthropic content-block ``type`` values stripped on normalize per FR-ING-7
# (tool traffic) and as non-durable noise (``thinking``).
_TOOL_BLOCK_TYPES = frozenset({"tool_use", "tool_result", "server_tool_use"})
_DROPPED_BLOCK_TYPES = _TOOL_BLOCK_TYPES | frozenset({"thinking", "redacted_thinking"})

# The directory under CLAUDE_CONFIG_DIR that holds per-project transcript dirs.
PROJECTS_DIRNAME = "projects"


def session_id_from_path(path: str | Path) -> str:
    """Derive the session id from a transcript path (the filename stem).

    Claude Code names each transcript ``<session-id>.jsonl``; the stem is the
    session id (FR-ING-2). Used as a fallback when a line omits ``sessionId``.
    """

    return Path(path).stem


def derive_project(path: str | Path, *, cwd: str | None = None) -> str:
    """Derive the ``project`` field for an event (FR-ING-2).

    Claude Code encodes the working directory into the per-project transcript
    directory name (e.g. ``-var-home-op-Projects-rust-cli``). The most reliable
    project label is the basename of the record's ``cwd`` when present; otherwise
    fall back to the basename of the encoded transcript-directory name.

    Parameters
    ----------
    path:
        The transcript file path (``.../projects/<encoded-dir>/<session>.jsonl``).
    cwd:
        The ``cwd`` field from the transcript line, if available — preferred
        because it is the literal working directory, not the path-encoded form.
    """

    if cwd:
        # A subagent/workflow may run in a git worktree at
        # ``<project>/.claude/worktrees/<id>`` — the basename there is the opaque
        # worktree id, so roll the scope up to ``<project>`` (FR-EXT-3: never an
        # opaque ``project:agent-XXXX`` scope).
        cwd_str = str(cwd)
        marker = "/.claude/worktrees/"
        if marker in cwd_str:
            cwd_str = cwd_str.split(marker, 1)[0]
        name = Path(cwd_str).name
        if name:
            return name

    p = Path(path)
    # The parent directory is the path-encoded project dir; its trailing segment
    # is the best available project label when no cwd is present.
    encoded = p.parent.name
    if not encoded:
        return p.stem
    # Claude Code replaces path separators with '-'; the final segment after the
    # last separator is the leaf working-directory name.
    leaf = encoded.rstrip("-").rsplit("-", 1)[-1]
    return leaf or encoded


# Subdirectory under a session dir that holds subagent / workflow transcripts.
# A transcript living under …/<encoded-cwd>/<session>/subagents/… is a subagent
# or workflow run of that parent session and MUST roll up to the parent project
# (FR-EXT-3 no opaque project:agent-XXXX scope).
_SUBAGENTS_DIRNAME = "subagents"
# A workflow segment id prefix (…/subagents/workflows/wf_<id>/agent-<id>.jsonl).
_WORKFLOW_SEGMENT_PREFIX = "wf_"


def decode_project_dirname(encoded: str) -> str:
    """Decode a Claude Code path-encoded project dir to a friendly project name.

    Claude Code names each per-project transcript dir by replacing the path
    separators of the working directory with ``-`` (e.g.
    ``-var-home-op-Projects-rust-cli`` for ``/var/home/op/Projects/rust-cli``).
    The friendly project name is the *last path component* of the decoded cwd —
    here ``rust-cli``.

    Because both the path separator and a literal hyphen inside a directory name
    both encode to ``-``, the decoding is lossy and the leaf cannot always be
    recovered exactly; this returns the trailing ``-``-separated segment, which
    is the best available label and matches what :func:`derive_project` produced.
    """

    leaf = encoded.strip("-").rsplit("-", 1)[-1]
    return leaf or encoded.strip("-") or encoded


def _project_dir_for_transcript(path: Path, projects_dirname: str) -> Path | None:
    """Return the top-level ``<projects>/<encoded-cwd>`` dir for a transcript.

    Walks up from the transcript file to find the ``<projects_dirname>`` ancestor
    and returns the immediate child of it on the path — the encoded-cwd project
    dir — regardless of how deep the transcript lives (a top-level session
    ``…/<encoded-cwd>/<session>.jsonl`` or a subagent
    ``…/<encoded-cwd>/<session>/subagents/…/agent-*.jsonl`` both resolve to the
    same ``<encoded-cwd>`` dir). Returns ``None`` if no ``projects`` ancestor is
    on the path.
    """

    parts = path.parts
    for i, part in enumerate(parts):
        if part == projects_dirname and i + 1 < len(parts):
            return Path(*parts[: i + 2])
    return None


def derive_scope_from_transcript(
    path: str | Path,
    settings: Settings | None = None,
    *,
    cwd: str | None = None,
) -> Scope:
    """Map a Claude Code transcript path to its hierarchical :class:`Scope` (FR-EXT-3).

    The parent PROJECT is the top-level ``$CLAUDE_CONFIG_DIR/projects/<encoded-cwd>``
    directory, DECODED to a friendly name (the last path component of the encoded
    cwd). The literal ``cwd`` from the transcript line is preferred when given
    (it is the real working directory, not the lossy encoded form).

    SUBAGENT / WORKFLOW transcripts live UNDER a session's dir
    (``…/<encoded-cwd>/<session>/subagents/…``) and ROLL UP to that same parent
    project — they never get an opaque ``project:agent-XXXX`` scope. A
    workflow segment (``…/subagents/workflows/wf_<id>/…``) is rolled up as an
    optional sub-segment (``project:<name>/wf_<id>``) when
    ``ScopeSettings.subagent_subsegments`` is enabled (default off → roll up to
    the bare ``project:<name>``), so subagent memories compose with the project
    via the ancestor chain (no-leak: still under the project, never a sibling).
    """

    from mnemozine.config import get_settings  # local import to avoid a cycle

    resolved = settings or get_settings()
    scope_cfg = resolved.scope
    p = Path(path)

    # 1. Find the encoded-cwd project dir (the immediate child of `projects/`).
    project_dir = _project_dir_for_transcript(p, PROJECTS_DIRNAME)

    # 2. Project name: the literal cwd leaf wins; else decode the encoded dir.
    project_name: str | None = None
    if cwd:
        # Strip a git-worktree suffix (``<project>/.claude/worktrees/<id>``) so a
        # subagent/workflow that ran in a worktree rolls up to ``<project>``, not
        # the opaque worktree id (FR-EXT-3: never a ``project:agent-XXXX`` scope).
        cwd_str = str(cwd)
        marker = "/.claude/worktrees/"
        if marker in cwd_str:
            cwd_str = cwd_str.split(marker, 1)[0]
        leaf = Path(cwd_str).name
        if leaf:
            project_name = leaf
    if project_name is None and project_dir is not None:
        project_name = decode_project_dirname(project_dir.name)
    if project_name is None:
        # No projects/ ancestor and no cwd — fall back to the parent dir name.
        project_name = decode_project_dirname(p.parent.name) if p.parent.name else p.stem

    scope = Scope.project(project_name)

    # 3. Roll a subagent/workflow transcript up under the same project. When
    #    sub-segmenting is enabled, attach the workflow id as a sub-segment so it
    #    composes with (and never leaks across) the project via the ancestor chain.
    if scope_cfg.subagent_subsegments and project_dir is not None:
        rel = p.relative_to(project_dir).parts if _is_relative_to(p, project_dir) else ()
        if _SUBAGENTS_DIRNAME in rel:
            wf = next(
                (seg for seg in rel if seg.startswith(_WORKFLOW_SEGMENT_PREFIX)), None
            )
            if wf:
                scope = scope.child(wf)
    return scope


def _is_relative_to(path: Path, other: Path) -> bool:
    """``Path.is_relative_to`` shim (stable across the supported Python range)."""

    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _parse_timestamp(value: Any) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating a trailing ``Z`` and naive input."""

    if isinstance(value, str) and value:
        text = value.replace("Z", "+00:00") if value.endswith("Z") else value
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return datetime.now(UTC)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    return datetime.now(UTC)


def _role_from(value: Any) -> Role | None:
    """Map an Anthropic message role to a :class:`Role`, or ``None`` if unknown."""

    if value == "user":
        return Role.USER
    if value == "assistant":
        return Role.ASSISTANT
    if value == "tool":
        return Role.TOOL
    return None


def _extract_text(content: Any, *, strip_tool_calls: bool) -> tuple[str, list[dict[str, Any]]]:
    """Flatten Anthropic message ``content`` into ``(text, tool_calls)``.

    ``content`` may be a plain string or a list of typed blocks. ``thinking`` and
    tool blocks are dropped from the text. When ``strip_tool_calls`` is ``False``
    the raw ``tool_use`` blocks are returned separately so the caller can attach
    them to ``IngestEvent.tool_calls`` (still never inlined into ``content``);
    when ``True`` they are discarded entirely (FR-ING-7).
    """

    if isinstance(content, str):
        return content.strip(), []

    if not isinstance(content, list):
        return "", []

    parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            # A bare string inside a content list is occasionally seen.
            if isinstance(block, str) and block.strip():
                parts.append(block.strip())
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        elif btype == "tool_use" or btype == "server_tool_use":
            if not strip_tool_calls:
                tool_calls.append(block)
        elif btype in _DROPPED_BLOCK_TYPES:
            # tool_result / tool_use / thinking — always stripped from text.
            continue
        # Unknown block types contribute nothing to durable memory.
    return "\n".join(parts).strip(), tool_calls


def parse_transcript_line(
    raw: str | dict[str, Any],
    *,
    path: str | Path,
    strip_tool_calls: bool = True,
    project: str | None = None,
    session_id: str | None = None,
) -> IngestEvent | None:
    """Parse one Claude Code JSONL line into an :class:`IngestEvent`.

    Returns ``None`` for non-conversational lines (bookkeeping records), malformed
    JSON, an unparseable role, or a turn whose only content was tool traffic that
    got stripped (so chunks are not padded with empty events).

    Parameters
    ----------
    raw:
        A JSON line (``str``) or an already-decoded record (``dict``).
    path:
        The transcript path, used to derive ``project``/``session_id`` defaults
        and to record the raw path in metadata (FR-ING-2).
    strip_tool_calls:
        Honor ``IngestSettings.strip_tool_calls`` (FR-ING-7). When ``True`` the
        emitted event carries ``tool_calls=None``; when ``False`` raw
        ``tool_use`` blocks are attached to ``tool_calls`` (never to ``content``).
    project / session_id:
        Optional overrides; when omitted they are derived from the line/path.
    """

    if isinstance(raw, str):
        line = raw.strip()
        if not line:
            return None
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None
    else:
        record = raw

    if not isinstance(record, dict):
        return None

    if record.get("type") not in _CONVERSATIONAL_TYPES:
        return None

    message = record.get("message")
    if not isinstance(message, dict):
        return None

    role = _role_from(message.get("role"))
    if role is None:
        return None

    text, tool_calls = _extract_text(
        message.get("content"), strip_tool_calls=strip_tool_calls
    )
    if not text:
        # The turn was entirely tool traffic / thinking — nothing durable left.
        return None

    resolved_session = session_id or record.get("sessionId") or session_id_from_path(path)
    cwd = record.get("cwd")
    resolved_project = project or derive_project(path, cwd=cwd if isinstance(cwd, str) else None)

    metadata: dict[str, Any] = {"raw_path": str(path)}
    if isinstance(cwd, str) and cwd:
        metadata["cwd"] = cwd
    git_branch = record.get("gitBranch")
    if isinstance(git_branch, str) and git_branch:
        metadata["git_branch"] = git_branch
    model = message.get("model")
    if isinstance(model, str) and model:
        metadata["model"] = model
    version = record.get("version")
    if isinstance(version, str) and version:
        metadata["cc_version"] = version

    return IngestEvent(
        source=Source.CLAUDE_CODE,
        project=str(resolved_project),
        session_id=str(resolved_session),
        timestamp=_parse_timestamp(record.get("timestamp")),
        role=role,
        content=text,
        tool_calls=tool_calls or None,
        metadata=metadata,
    )


def parse_transcript_lines(
    lines: Iterable[str | dict[str, Any]],
    *,
    path: str | Path,
    strip_tool_calls: bool = True,
    project: str | None = None,
    session_id: str | None = None,
) -> Iterator[IngestEvent]:
    """Parse many transcript lines, skipping the ones that yield ``None``.

    A convenience over :func:`parse_transcript_line` for a whole transcript:
    bookkeeping lines, malformed JSON, and tool-only turns are silently dropped,
    so the result is exactly the conversational events in file order.
    """

    for raw in lines:
        event = parse_transcript_line(
            raw,
            path=path,
            strip_tool_calls=strip_tool_calls,
            project=project,
            session_id=session_id,
        )
        if event is not None:
            yield event


def read_transcript(
    path: str | Path,
    *,
    strip_tool_calls: bool = True,
    project: str | None = None,
) -> list[IngestEvent]:
    """Read and parse a whole transcript file into ordered events (FR-ING-2/6).

    Reads the JSONL file at ``path`` and returns its conversational events. Missing
    files yield an empty list (a transcript may be deleted mid-tail by the 30-day
    cleanup, R4); the caller decides whether that is an error.
    """

    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError):
        return []
    return list(
        parse_transcript_lines(
            text.splitlines(),
            path=p,
            strip_tool_calls=strip_tool_calls,
            project=project,
        )
    )
