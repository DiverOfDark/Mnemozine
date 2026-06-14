"""Graph explorer route tests (PRD §4.4) — offline against fakes.

Asserts the scoped subgraph carries entity nodes + weighted structural edges and
idea-seed memory nodes linked to their entities, and that the cross-reference
overlay (when present) is flagged ``is_crossref`` with a non-empty reason.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_graph_returns_entity_nodes_and_weighted_edges(client: TestClient) -> None:
    resp = client.get("/api/graph", params={"include_crossrefs": "false"})
    assert resp.status_code == 200
    body = resp.json()
    entity_nodes = [n for n in body["nodes"] if n["kind"] == "entity"]
    labels = {n["label"] for n in entity_nodes}
    assert {"rust", "error-handling", "tokio"} <= labels
    # Structural edges carry a weight + active flag and are not cross-refs.
    structural = [e for e in body["edges"] if not e["is_crossref"]]
    assert structural
    rust_err = next(
        (e for e in structural if e["relation"] == "relates_to"), None
    )
    assert rust_err is not None
    assert rust_err["weight"] == 0.8
    assert rust_err["active"] is True


def test_graph_entity_type_filter(client: TestClient) -> None:
    body = client.get(
        "/api/graph", params={"entity_type": "library", "include_crossrefs": "false"}
    ).json()
    entity_nodes = [n for n in body["nodes"] if n["kind"] == "entity"]
    assert {n["label"] for n in entity_nodes} == {"tokio"}


def test_graph_includes_idea_seed_nodes(client: TestClient) -> None:
    body = client.get("/api/graph", params={"include_crossrefs": "false"}).json()
    idea_nodes = [n for n in body["nodes"] if n["kind"] == "idea_seed"]
    assert idea_nodes, "the archived idea_seed memory should be a graph node"
    # The idea-seed node links to the entities it mentions via `mentions` edges.
    mention_edges = [e for e in body["edges"] if e["relation"] == "mentions"]
    assert mention_edges


def test_graph_center_on_entity(client: TestClient) -> None:
    body = client.get(
        "/api/graph", params={"entity": "rust", "depth": 1, "include_crossrefs": "false"}
    ).json()
    entity_labels = {n["label"] for n in body["nodes"] if n["kind"] == "entity"}
    # rust + its 1-hop neighbors (error-handling, tokio) are present.
    assert "rust" in entity_labels
    assert entity_labels <= {"rust", "error-handling", "tokio"}


def test_graph_crossref_overlay_edges_carry_reason(client: TestClient) -> None:
    # The overlay is best-effort; when crossref edges are present they must carry a
    # non-empty reason and be flagged is_crossref (FR-RET-6).
    body = client.get(
        "/api/graph", params={"include_crossrefs": "true"}
    ).json()
    for edge in body["edges"]:
        if edge["is_crossref"]:
            assert edge["reason"], "a cross-ref edge must carry a human-readable reason"
