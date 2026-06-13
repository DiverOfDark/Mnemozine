"""Retrieval & delivery layer (FR-RET-*).

This package implements the FR-RET-* requirements on top of the
:class:`mnemozine.interfaces.StorageBackend` Protocol — never a concrete sibling
module — so it works identically against Graphiti/FalkorDB and the offline test
fakes.

Public surface:

* :class:`ScopedRetriever` — the :class:`mnemozine.interfaces.Retriever`
  implementation: FR-RET-2 scoped retrieve (compose current project + global +
  entity-linked neighborhood, then semantic search within that subset),
  FR-RET-3/5 ``build_index`` (compact ~500-token injection), FR-RET-4 ``recall``.
* :func:`detect_context` — derive a :class:`~mnemozine.interfaces.RetrievalContext`
  from cwd / ``Cargo.toml`` / ``package.json`` / git-remote / recent turns
  (FR-RET-3), offline.
* :func:`session_start_injection` / :func:`mid_session_injection` — the
  FR-RET-3 / FR-RET-5 hook helpers.
* :func:`build_mcp_server` / :func:`run_server` — the single MCP server exposing
  ``recall`` (FR-RET-1 / FR-RET-4). The ``mcp`` SDK is imported lazily inside
  :func:`build_mcp_server`, so importing this package does not require ``mcp``.
* :func:`main` — the ``mnemozine-mcp`` console-script entrypoint.

``estimate_tokens`` / ``render_index`` (the budget machinery) are exported for
the §9 budget-enforcement tests.
"""

from __future__ import annotations

from mnemozine.retrieval.budget import (
    IndexParts,
    estimate_tokens,
    render_index,
)
from mnemozine.retrieval.context import detect_context, project_from_git_remote
from mnemozine.retrieval.injection import (
    mid_session_injection,
    session_start_injection,
)
from mnemozine.retrieval.retriever import ScopedRetriever

__all__ = [
    "ScopedRetriever",
    "detect_context",
    "project_from_git_remote",
    "session_start_injection",
    "mid_session_injection",
    "estimate_tokens",
    "render_index",
    "IndexParts",
    "build_mcp_server",
    "run_server",
    "main",
]


def build_mcp_server(retriever: ScopedRetriever, **kwargs: object) -> object:
    """Lazy re-export of :func:`mnemozine.retrieval.server.build_mcp_server`.

    Deferred import so ``import mnemozine.retrieval`` never pulls in the ``mcp``
    SDK (kept optional/heavy). See that function for the full docstring.
    """

    from mnemozine.retrieval.server import build_mcp_server as _build

    return _build(retriever, **kwargs)  # type: ignore[arg-type]


def run_server(retriever: ScopedRetriever, **kwargs: object) -> None:
    """Lazy re-export of :func:`mnemozine.retrieval.server.run`."""

    from mnemozine.retrieval.server import run as _run

    _run(retriever, **kwargs)  # type: ignore[arg-type]


def main() -> None:
    """Console-script entrypoint for ``mnemozine-mcp`` (FR-RET-1).

    Builds a :class:`ScopedRetriever` over the wired storage backend and serves
    the single MCP server. Storage wiring lives in the integration pass
    (:class:`mnemozine.app.Container.build_storage`); this entrypoint composes a
    retriever over it and runs the server. It imports the heavy deps (``mcp``,
    the backend) lazily so the module stays importable in test/CI without them.
    """

    import asyncio

    from mnemozine.app import Container
    from mnemozine.retrieval.server import run as _run

    container = Container.from_env()
    settings = container.settings
    # build_storage() is async (it opens the FalkorDB connection); build the
    # retriever over the connected backend before handing it to the blocking
    # server.run.
    retriever = asyncio.run(container.build_retriever())
    _run(retriever, settings=settings)  # type: ignore[arg-type]
