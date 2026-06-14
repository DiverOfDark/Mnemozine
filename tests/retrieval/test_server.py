"""FR-RET-1 / FR-RET-4 MCP server tests against the offline fakes.

These build the real FastMCP server over an ``InMemoryStorage``-backed
``ScopedRetriever`` and invoke the registered tools through the SDK's tool
manager — verifying ``recall`` (FR-RET-4) returns the right JSON-projected,
scope-composed memory with no live FalkorDB.
"""

from __future__ import annotations

import json

from mnemozine.config import Settings
from mnemozine.retrieval.retriever import ScopedRetriever
from mnemozine.retrieval.server import RecalledMemory, _parse_scope, build_mcp_server
from mnemozine.schema.models import (
    MemoryUnit,
    Provenance,
    Scope,
)
from tests.conftest import InMemoryStorage


def _mem(
    content: str,
    scope: Scope,
    entities: list[str],
    *,
    category: str = "fact",
    cross_ref_candidate: bool = False,
) -> MemoryUnit:
    """Build a MemoryUnit on the category-split contract (no legacy ``type``)."""

    return MemoryUnit(
        content=content,
        scope=scope,
        category=category,
        cross_ref_candidate=cross_ref_candidate,
        entities=entities,
        confidence=0.9,
        provenance=Provenance(source="claude_code", session_id="sess-1"),
    )


async def _seed(storage: InMemoryStorage) -> dict[str, MemoryUnit]:
    pref = _mem(
        "Prefers thiserror over anyhow for rust error handling",
        Scope.global_(),
        ["rust", "error-handling"],
        category="preference",
    )
    fact_here = _mem(
        "rust-cli pins tokio 1.38",
        Scope.project("rust-cli"),
        ["rust", "tokio"],
        category="decision",
    )
    fact_other = _mem(
        "other-proj uses postgres 16",
        Scope.project("other-proj"),
        ["postgres"],
        category="decision",
    )
    for m in (pref, fact_here, fact_other):
        await storage.upsert_memory(m)
    return {"pref": pref, "fact_here": fact_here, "fact_other": fact_other}


def _make_server(storage: InMemoryStorage, settings: Settings):
    retriever = ScopedRetriever(storage, settings=settings, default_project="rust-cli")
    return build_mcp_server(retriever, settings=settings)


def test_parse_scope_variants() -> None:
    assert _parse_scope(None) is None
    assert _parse_scope("") is None
    assert _parse_scope("global").is_global
    assert _parse_scope("project:foo").as_str() == "project:foo"
    # Bare project id convenience.
    assert _parse_scope("foo").as_str() == "project:foo"


def test_recalled_memory_projection() -> None:
    from mnemozine.interfaces import RetrievedMemory

    m = _mem(
        "x content",
        Scope.global_(),
        ["a"],
        category="preference",
        cross_ref_candidate=True,
    )
    rm = RecalledMemory.from_retrieved(RetrievedMemory(memory=m, score=0.5))
    assert rm.content == "x content"
    # The category split: the wire view exposes the free-form category + the
    # cross_ref_candidate flag instead of the dropped MemoryType.
    assert rm.category == "preference"
    assert rm.cross_ref_candidate is True
    assert rm.scope == "global"
    assert rm.source == "claude_code"


async def test_server_exposes_recall_tool(settings: Settings) -> None:
    storage = InMemoryStorage()
    await _seed(storage)
    server = _make_server(storage, settings)
    tools = await server.list_tools()
    names = {t.name for t in tools}
    assert "recall" in names  # FR-RET-1 / FR-RET-4
    assert "session_start_index" in names  # FR-RET-3
    assert "mid_session_index" in names  # FR-RET-5


def _extract_payload(result) -> list[dict]:
    """Pull the structured/JSON payload out of a FastMCP call_tool result.

    FastMCP returns a (content, structured) tuple; the structured value carries
    the tool's return. Fall back to parsing the text content if needed.
    """

    content, structured = result
    if structured is not None:
        # FastMCP wraps non-dict returns under a 'result' key.
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        return structured
    # Fallback: parse the first text content block.
    for block in content:
        text = getattr(block, "text", None)
        if text:
            return json.loads(text)
    return []


async def test_recall_tool_composes_scope_and_excludes_other_project(
    settings: Settings,
) -> None:
    storage = InMemoryStorage()
    seeded = await _seed(storage)
    server = _make_server(storage, settings)

    result = await server.call_tool("recall", {"query": "rust tokio thiserror postgres"})
    payload = _extract_payload(result)
    contents = {row["content"] for row in payload}
    # Current project + global compose.
    assert seeded["pref"].content in contents
    assert seeded["fact_here"].content in contents
    # Other project's fact must not leak (no-leak §9 / FR-RET-2).
    assert seeded["fact_other"].content not in contents


async def test_recall_tool_explicit_global_scope(settings: Settings) -> None:
    storage = InMemoryStorage()
    seeded = await _seed(storage)
    server = _make_server(storage, settings)

    result = await server.call_tool("recall", {"query": "rust thiserror tokio", "scope": "global"})
    payload = _extract_payload(result)
    contents = {row["content"] for row in payload}
    assert seeded["pref"].content in contents
    # Project facts excluded under an explicit global scope.
    assert seeded["fact_here"].content not in contents


async def test_recall_tool_records_access(settings: Settings) -> None:
    storage = InMemoryStorage()
    seeded = await _seed(storage)
    server = _make_server(storage, settings)
    await server.call_tool("recall", {"query": "rust thiserror"})
    assert storage.memories[seeded["pref"].id].access_count >= 1


async def test_session_start_index_tool_under_budget(tmp_path, settings: Settings) -> None:
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "rust-cli"\n[dependencies]\nthiserror = "1"\n',
        encoding="utf-8",
    )
    storage = InMemoryStorage()
    await _seed(storage)
    server = _make_server(storage, settings)
    result = await server.call_tool(
        "session_start_index",
        {"cwd": str(tmp_path), "git_remote": "git@github.com:op/rust-cli.git"},
    )
    content, structured = result
    assert structured is not None
    assert structured["token_estimate"] <= settings.inject.token_budget
