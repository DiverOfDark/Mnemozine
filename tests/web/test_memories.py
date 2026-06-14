"""Memories list + detail route tests (PRD §4.2/§4.3) — offline against fakes.

Covers the table filters (category / scope / tier / entity / active-vs-superseded
/ source / free-text), the category-facets + scope-tree discovery endpoints,
pagination, and the detail view's provenance + validity window + supersession
chain (the signature temporal feature). The old fixed ``type`` enum filter is now
the free-form ``category`` filter.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


def _ids(items: list[dict[str, Any]]) -> set[str]:
    return {i["id"] for i in items}


def test_list_returns_all_seeded_memories(client: TestClient) -> None:
    resp = client.get("/api/memories")
    assert resp.status_code == 200
    body = resp.json()
    assert body["page"]["total"] == 4
    assert _ids(body["items"]) == {
        "mem-pref-current",
        "mem-pref-stale",
        "mem-fact-tokio",
        "mem-idea-cli",
    }
    # Every row carries the explicit `active` boolean (never derived by the UI).
    assert all("active" in i for i in body["items"])


def test_list_filter_by_category(client: TestClient) -> None:
    resp = client.get("/api/memories", params={"category": "preference"})
    assert resp.status_code == 200
    assert _ids(resp.json()["items"]) == {"mem-pref-current", "mem-pref-stale"}


def test_list_filter_by_category_is_case_insensitive(client: TestClient) -> None:
    # Categories are normalized (lowercased/trimmed); the filter normalizes too.
    resp = client.get("/api/memories", params={"category": "  Preference "})
    assert _ids(resp.json()["items"]) == {"mem-pref-current", "mem-pref-stale"}


def test_list_item_carries_category_and_scope(client: TestClient) -> None:
    items = {i["id"]: i for i in client.get("/api/memories").json()["items"]}
    # The free-form category chip + the persisted scope path are on every row.
    assert items["mem-fact-tokio"]["category"] == "decision"
    assert items["mem-fact-tokio"]["scope"] == "project:rust-cli"
    assert items["mem-fact-tokio"]["scope_decision"] == "project"
    assert items["mem-idea-cli"]["category"] == "idea"
    assert items["mem-idea-cli"]["cross_ref_candidate"] is True
    assert items["mem-pref-current"]["scope"] == "global"
    assert items["mem-pref-current"]["scope_decision"] == "global"


def test_list_filter_by_tier_archive(client: TestClient) -> None:
    resp = client.get("/api/memories", params={"tier": "archive"})
    assert _ids(resp.json()["items"]) == {"mem-idea-cli"}


def test_list_filter_by_scope_project(client: TestClient) -> None:
    resp = client.get("/api/memories", params={"scope": "project:rust-cli"})
    assert _ids(resp.json()["items"]) == {"mem-fact-tokio"}


def test_list_filter_by_bare_project_id(client: TestClient) -> None:
    # The contract allows a bare project id as a convenience scope form.
    resp = client.get("/api/memories", params={"scope": "rust-cli"})
    assert _ids(resp.json()["items"]) == {"mem-fact-tokio"}


def test_list_filter_active_only(client: TestClient) -> None:
    resp = client.get("/api/memories", params={"active": "true"})
    ids = _ids(resp.json()["items"])
    assert "mem-pref-stale" not in ids  # superseded excluded
    assert "mem-pref-current" in ids


def test_list_filter_superseded_only(client: TestClient) -> None:
    resp = client.get("/api/memories", params={"active": "false"})
    assert _ids(resp.json()["items"]) == {"mem-pref-stale"}


def test_list_filter_by_source(client: TestClient) -> None:
    resp = client.get("/api/memories", params={"source": "openai"})
    assert _ids(resp.json()["items"]) == {"mem-fact-tokio"}


def test_list_filter_by_entity(client: TestClient) -> None:
    resp = client.get("/api/memories", params={"entity": "tokio"})
    assert _ids(resp.json()["items"]) == {"mem-fact-tokio", "mem-idea-cli"}


def test_list_free_text_search(client: TestClient) -> None:
    resp = client.get("/api/memories", params={"q": "tokio"})
    assert _ids(resp.json()["items"]) == {"mem-fact-tokio", "mem-idea-cli"}


def test_list_pagination(client: TestClient) -> None:
    first = client.get("/api/memories", params={"limit": 2, "offset": 0}).json()
    second = client.get("/api/memories", params={"limit": 2, "offset": 2}).json()
    assert first["page"]["total"] == 4
    assert len(first["items"]) == 2
    assert len(second["items"]) == 2
    # No overlap between pages.
    assert _ids(first["items"]).isdisjoint(_ids(second["items"]))


def test_list_ordered_newest_first(client: TestClient) -> None:
    items = client.get("/api/memories").json()["items"]
    valid_from = [i["valid_from"] for i in items]
    assert valid_from == sorted(valid_from, reverse=True)


def test_detail_carries_provenance_and_validity(client: TestClient) -> None:
    resp = client.get("/api/memories/mem-pref-current")
    assert resp.status_code == 200
    body = resp.json()
    assert body["provenance"]["source"] == "claude_code"
    assert body["provenance"]["session_id"] == "sess-1"
    assert body["provenance"]["raw_path"]
    assert body["validity"]["active"] is True
    assert body["validity"]["valid_to"] is None


def test_detail_supersession_chain(client: TestClient) -> None:
    # The current preference replaced the stale one: it appears in `supersedes`.
    current = client.get("/api/memories/mem-pref-current").json()
    assert "mem-pref-stale" in {s["memory_id"] for s in current["supersedes"]}
    assert current["superseded_by"] == []

    # The stale preference is superseded-by the current one (closed window).
    stale = client.get("/api/memories/mem-pref-stale").json()
    assert stale["validity"]["active"] is False
    assert stale["validity"]["valid_to"] is not None
    assert "mem-pref-current" in {s["memory_id"] for s in stale["superseded_by"]}


def test_detail_unknown_id_404(client: TestClient) -> None:
    resp = client.get("/api/memories/does-not-exist")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Category facets (the dynamic CATEGORY filter discovery surface)
# ---------------------------------------------------------------------------


def test_category_facets_lists_discovered_categories(client: TestClient) -> None:
    resp = client.get("/api/memories/facets/categories")
    assert resp.status_code == 200
    body = resp.json()
    by_cat = {f["category"]: f["count"] for f in body["facets"]}
    # The active categories in the seed (the superseded 'preference' is excluded
    # by list_categories, which counts active memories only).
    assert by_cat == {"preference": 1, "decision": 1, "idea": 1}
    assert body["total"] == 3


def test_category_facets_ordered_most_frequent_first(client: TestClient) -> None:
    facets = client.get("/api/memories/facets/categories").json()["facets"]
    counts = [f["count"] for f in facets]
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# Scope tree (the hierarchical SCOPE NAVIGATOR discovery surface)
# ---------------------------------------------------------------------------


def test_scope_tree_has_global_root_with_children(client: TestClient) -> None:
    resp = client.get("/api/memories/facets/scope-tree")
    assert resp.status_code == 200
    root = resp.json()["root"]
    assert root["segment"] == "global"
    assert root["path"] == "global"
    assert root["depth"] == 0
    # Three memories sit at global (current/stale prefs + the archived idea);
    # the project memory rolls up under the project child.
    assert root["count"] == 3
    assert root["total_count"] == 4
    child_paths = {c["path"] for c in root["children"]}
    assert "project:rust-cli" in child_paths


def test_scope_tree_project_node_counts(client: TestClient) -> None:
    root = client.get("/api/memories/facets/scope-tree").json()["root"]
    project = next(c for c in root["children"] if c["path"] == "project:rust-cli")
    assert project["segment"] == "rust-cli"
    assert project["depth"] == 1
    assert project["count"] == 1
    assert project["total_count"] == 1
    assert project["children"] == []
