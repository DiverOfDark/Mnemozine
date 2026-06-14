"""Eval read route tests (PRD §4.8) — offline against the §9 harness + fakes.

Covers the eval summary (runs the offline harness over the committed gold set) and
the F4 bootstrap-labeling queue (read from the shared in-process store).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from mnemozine.web.routes._bootstrap_state import bootstrap_store


def test_eval_summary_runs_harness(client: TestClient) -> None:
    resp = client.get("/api/eval")
    assert resp.status_code == 200
    body = resp.json()
    assert body["gold_set"]
    assert isinstance(body["passed"], bool)
    metric_names = {m["name"] for m in body["metrics"]}
    # The §9 harness reports its core metrics.
    assert "classifier_accuracy" in metric_names
    assert "injection_precision_at_k" in metric_names
    for m in body["metrics"]:
        assert "value" in m and "passed" in m


def test_bootstrap_queue_lists_candidates(client: TestClient) -> None:
    bootstrap_store.reset()
    resp = client.get("/api/eval/bootstrap")
    assert resp.status_code == 200
    candidates = resp.json()["candidates"]
    assert candidates, "the bootstrap queue self-seeds with candidates"
    for c in candidates:
        assert c["candidate_id"]
        assert c["label"] in {"unreviewed", "keep", "drop"}
        assert c["proposed_type"] in {"preference", "project_fact", "idea_seed"}


def test_bootstrap_read_reflects_shared_store_label(client: TestClient) -> None:
    # A label written through the shared store (as the write route would) is
    # reflected in the read route — proving they share one source of truth.
    bootstrap_store.reset()
    seeded = client.get("/api/eval/bootstrap").json()["candidates"]
    target_id = seeded[0]["candidate_id"]
    bootstrap_store.label(target_id, label="keep")

    refreshed = client.get("/api/eval/bootstrap").json()["candidates"]
    by_id = {c["candidate_id"]: c for c in refreshed}
    assert by_id[target_id]["label"] == "keep"
    bootstrap_store.reset()
