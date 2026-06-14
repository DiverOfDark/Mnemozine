"""Recall playground route tests (PRD §4.5) — offline against fakes.

Drives the real ``Retriever.recall`` + ``Retriever.build_index`` over the seeded
in-memory store and asserts the ranked results carry scores + a why-note and that
the SessionStart index preview is returned with the configured token budget.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_recall_returns_ranked_scored_results(client: TestClient) -> None:
    resp = client.post(
        "/api/recall",
        json={"query": "rust error handling", "top_k": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "rust error handling"
    assert body["results"], "recall should surface the rust preference"
    first = body["results"][0]
    assert "score" in first
    assert first["why"]  # a non-empty explanation
    # The active current preference surfaces; the superseded one does not.
    ids = {r["memory"]["id"] for r in body["results"]}
    assert "mem-pref-current" in ids
    assert "mem-pref-stale" not in ids


def test_recall_index_preview_present_with_budget(client: TestClient) -> None:
    body = client.post(
        "/api/recall",
        json={"query": "rust", "top_k": 5, "include_index_preview": True},
    ).json()
    preview = body["index_preview"]
    assert preview is not None
    assert preview["token_budget"] == 500  # inject.token_budget default
    assert preview["token_estimate"] <= preview["token_budget"]
    assert isinstance(preview["entity_tags"], list)


def test_recall_index_preview_omitted_when_not_requested(client: TestClient) -> None:
    body = client.post(
        "/api/recall",
        json={"query": "rust", "include_index_preview": False},
    ).json()
    assert body["index_preview"] is None


def test_recall_scope_echoed_back(client: TestClient) -> None:
    body = client.post(
        "/api/recall",
        json={"query": "tokio", "scope": "project:rust-cli"},
    ).json()
    assert body["scope"] == "project:rust-cli"
    ids = {r["memory"]["id"] for r in body["results"]}
    assert "mem-fact-tokio" in ids
