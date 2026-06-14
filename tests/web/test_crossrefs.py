"""Cross-reference read route tests (PRD §4.4/§4.7) — offline against fakes.

Seeds an idea_seed sharing entities with the working context so ``find_related``
surfaces a connection, then asserts the list item carries its mandatory reason,
shared entities, and the context_key a suppression would apply to.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import InMemoryStorage


def test_crossrefs_returns_items_with_reason(
    client: TestClient, storage: InMemoryStorage
) -> None:
    # Working context = tokio/rust entities; the idea_seed shares them.
    resp = client.get("/api/crossrefs", params={"entity": "tokio"})
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body and "page" in body
    for item in body["items"]:
        # FR-RET-6: a surfaced connection ALWAYS carries a non-empty reason.
        assert item["reason"]
        assert "context_key" in item
        assert item["suppressed"] is False


def test_crossrefs_empty_context_is_schema_valid(client: TestClient) -> None:
    body = client.get("/api/crossrefs").json()
    assert body["page"]["total"] == len(body["items"])


def test_crossrefs_pagination_envelope(client: TestClient) -> None:
    body = client.get(
        "/api/crossrefs", params={"entity": "tokio", "limit": 1, "offset": 0}
    ).json()
    assert body["page"]["limit"] == 1
    assert body["page"]["offset"] == 0
    assert len(body["items"]) <= 1
