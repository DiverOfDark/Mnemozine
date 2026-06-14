"""The single Mnemozine MCP server (FR-RET-1) exposing ``recall`` (FR-RET-4).

One MCP server serves memory to **all** agents — Claude Code and OpenAI/Hermes
agents alike (FR-RET-1). It is built on the official ``mcp`` Python SDK
(``FastMCP``) and exposes the on-demand :func:`recall` tool (FR-RET-4) so any
connected agent can pull consolidated full-detail memory when an injected index
hints at something worth chasing.

The server is deliberately *thin*: it owns transport + tool schema only, and
delegates every retrieval decision to a :class:`ScopedRetriever` (which depends
solely on the ``StorageBackend`` Protocol). This keeps FR-RET-2 scope
composition and the FR-RET-3/5 budget logic in one place and lets the server be
unit-tested against the offline fakes with no live FalkorDB.

Transport
---------
* ``stdio`` — for Claude Code's local MCP client (the default).
* ``streamable-http`` / ``sse`` — for networked agents (OpenAI/Hermes), bound to
  ``settings.mcp_host`` / ``settings.mcp_port``.

The integration pass wires a real :class:`ScopedRetriever` (over the Graphiti/
FalkorDB backend) and calls :func:`run` from the ``mnemozine-mcp`` console
script. See ``build_mcp_server`` for the construction seam used in tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import RetrievedMemory
from mnemozine.retrieval.context import detect_context
from mnemozine.retrieval.injection import (
    mid_session_injection,
    session_start_injection,
)
from mnemozine.retrieval.retriever import ScopedRetriever
from mnemozine.schema.models import Scope


@dataclass(slots=True)
class RecalledMemory:
    """A flat, JSON-serializable view of a recalled memory for an MCP client.

    The MCP wire surface is plain JSON, so we project the rich
    :class:`~mnemozine.schema.models.MemoryUnit` down to the fields an agent
    actually needs: the distilled content, its type/scope, the relevance score,
    the linked entities, confidence, and a provenance pointer for auditing.
    """

    id: str
    type: str
    content: str
    scope: str
    score: float
    confidence: float
    entities: list[str]
    source: str
    session_id: str

    @classmethod
    def from_retrieved(cls, retrieved: RetrievedMemory) -> RecalledMemory:
        m = retrieved.memory
        return cls(
            id=m.id,
            type=m.type.value,
            content=m.content,
            scope=m.scope.as_str(),
            score=round(retrieved.score, 6),
            confidence=m.confidence,
            entities=list(m.entities),
            source=m.provenance.source,
            session_id=m.provenance.session_id,
        )


def _parse_scope(scope: str | None) -> Scope | None:
    """Parse an optional MCP ``scope`` argument into a :class:`Scope`.

    Accepts the persisted string form (``global`` / ``project:<id>``) or a bare
    project id (treated as ``project:<id>``). ``None`` / empty -> default
    composed scope (handled by the retriever). Raises ``ValueError`` on a
    malformed non-empty value so the MCP client gets a clear error.
    """

    if scope is None:
        return None
    value = scope.strip()
    if not value:
        return None
    if value == "global" or value.startswith("project:"):
        return Scope.parse(value)
    # Bare project id convenience.
    return Scope.project(value)


def build_mcp_server(
    retriever: ScopedRetriever,
    *,
    settings: Settings | None = None,
    streamable_http_path: str | None = None,
) -> Any:
    """Build the FastMCP server exposing the ``recall`` tool (FR-RET-1/4).

    Constructed with a ready :class:`ScopedRetriever` so the same builder works
    in tests (over ``InMemoryStorage``) and in production (over Graphiti/
    FalkorDB). The ``mcp`` SDK is imported lazily so importing this module — and
    therefore the whole ``mnemozine.retrieval`` package — never requires ``mcp``
    to be installed (keeps schema/contract imports light).

    Tools exposed:

    * ``recall(query, scope=None, top_k=10)`` — FR-RET-4 on-demand recall.
    * ``session_start_index(cwd=None, git_remote=None, recent_text=None)`` —
      FR-RET-3 proactive index (also callable as an MCP tool so non-hook agents
      can request the same compact index).
    * ``mid_session_index(prompt, project=None)`` — FR-RET-5 finer-grained index.
    """

    from mcp.server.fastmcp import FastMCP

    cfg = settings or get_settings()
    # When mounting the streamable-HTTP app into another ASGI app (the all-in-one
    # ``run_all`` path), the sub-app's own route should be at "/" and the *mount*
    # supplies the "/mcp" prefix — otherwise the route lands at "/mcp/mcp". For the
    # standalone HTTP server the default "/mcp" is correct, so this is opt-in.
    extra: dict[str, Any] = {}
    if streamable_http_path is not None:
        extra["streamable_http_path"] = streamable_http_path
    server = FastMCP(
        name="mnemozine",
        instructions=(
            "Mnemozine unified memory. Call recall(query, scope?) to pull "
            "consolidated full-detail memory (preferences, project facts, idea "
            "seeds) across all past sessions when you need detail beyond the "
            "injected SessionStart index. scope is optional: omit for the "
            "current project + global, or pass 'global' / 'project:<id>'."
        ),
        host=cfg.mcp_host,
        port=cfg.mcp_port,
        **extra,
    )

    @server.tool(
        name="recall",
        description=(
            "On-demand recall of consolidated memory (FR-RET-4). Returns the "
            "current, deduped memory units relevant to `query`, scoped to the "
            "current project + global preferences by default. Pass `scope` as "
            "'global', 'project:<id>', or a bare project id to narrow."
        ),
    )
    async def recall(query: str, scope: str | None = None, top_k: int = 10) -> list[dict[str, Any]]:
        parsed = _parse_scope(scope)
        results = await retriever.recall(query, parsed, top_k=top_k)
        return [asdict(RecalledMemory.from_retrieved(r)) for r in results]

    @server.tool(
        name="session_start_index",
        description=(
            "Proactive SessionStart memory index (FR-RET-3): a compact, "
            "token-budgeted (~500) advisory summary of relevant preferences, "
            "project facts and possibly-related ideas for the current working "
            "context. Detail is pulled via recall()."
        ),
    )
    async def session_start_index(
        cwd: str | None = None,
        git_remote: str | None = None,
        recent_text: str | None = None,
    ) -> dict[str, Any]:
        index = await session_start_injection(
            retriever,
            cwd=cwd,
            git_remote=git_remote,
            recent_text=recent_text,
        )
        return {
            "text": index.text,
            "token_estimate": index.token_estimate,
            "preference_count": index.preference_count,
            "project_fact_count": index.project_fact_count,
            "idea_seed_hints": index.idea_seed_hints,
            "entity_tags": index.entity_tags,
        }

    @server.tool(
        name="mid_session_index",
        description=(
            "Finer-grained mid-session memory index (FR-RET-5) for a new prompt, "
            "drawn from the same ~500-token budget envelope but smaller. Use as "
            "the conversation moves into a new topic."
        ),
    )
    async def mid_session_index(prompt: str, project: str | None = None) -> dict[str, Any]:
        index = await mid_session_injection(retriever, prompt, project=project, settings=cfg)
        return {
            "text": index.text,
            "token_estimate": index.token_estimate,
            "preference_count": index.preference_count,
            "project_fact_count": index.project_fact_count,
            "idea_seed_hints": index.idea_seed_hints,
            "entity_tags": index.entity_tags,
        }

    return server


def build_mcp_http_app(
    retriever: ScopedRetriever,
    *,
    settings: Settings | None = None,
    streamable_http_path: str | None = None,
) -> tuple[Any, Any]:
    """Build the FastMCP server + its streamable-HTTP ASGI app for mounting.

    Returns ``(server, asgi_app)`` where ``asgi_app`` is the Starlette app
    produced by ``FastMCP.streamable_http_app()``. Its single route is the server's
    ``streamable_http_path`` (``/mcp`` by default for the standalone HTTP server;
    pass ``streamable_http_path="/"`` so the route is at the sub-app root and the
    parent ``Mount("/mcp", ...)`` supplies the prefix — the seam the all-in-one
    ``run_all`` uses to serve the WebUI + MCP from one port, resolving the 8765
    web/MCP clash).

    Calling ``streamable_http_app()`` also lazily constructs the server's
    ``StreamableHTTPSessionManager``; that manager must be *run* for the transport
    to work. When mounting into another app, the caller must drive
    ``server.session_manager.run()`` for the lifetime of the parent app (see
    :func:`mnemozine.app._build_web_app`), because the mounted sub-app's own
    lifespan is not invoked by a Starlette ``Mount``.
    """

    server = build_mcp_server(
        retriever, settings=settings, streamable_http_path=streamable_http_path
    )
    asgi_app = server.streamable_http_app()
    return server, asgi_app


def run(
    retriever: ScopedRetriever,
    *,
    transport: str = "stdio",
    settings: Settings | None = None,
) -> None:
    """Run the MCP server (blocking) over the given transport (FR-RET-1).

    ``transport`` is one of ``stdio`` (Claude Code local default),
    ``streamable-http`` or ``sse`` (networked OpenAI/Hermes agents). The
    integration pass calls this from the ``mnemozine-mcp`` console script with a
    retriever wired over the real backend.
    """

    server = build_mcp_server(retriever, settings=settings)
    server.run(transport=transport)


__all__ = [
    "RecalledMemory",
    "build_mcp_server",
    "build_mcp_http_app",
    "run",
    "detect_context",
]
