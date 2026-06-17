"""CoMentionJob tests — weighted entity-entity co-mention layer.

Covers Protocol conformance, the pure weighting/cap helpers (offline, no storage),
the applied ``run`` pass against the conftest ``InMemoryStorage`` fake (derives
pairs from the mention layer, hub down-weight, degree cap), and the FR-MNT-5
idempotency property (a re-run upserts the SAME edges and grows nothing — the
upsert re-asserts weight rather than summing).
"""

from __future__ import annotations

import math

import pytest

from mnemozine.config import GraphSettings, Settings
from mnemozine.interfaces import MaintenanceJob
from mnemozine.maintenance.co_mention import CoMentionJob, co_mention_weight
from mnemozine.schema.models import Entity, MemoryUnit, Provenance, Scope
from tests.conftest import InMemoryStorage


def _mem(content: str, *, entities: list[str], mid: str) -> MemoryUnit:
    return MemoryUnit(
        id=mid,
        content=content,
        scope=Scope.global_(),
        category="fact",
        entities=entities,
        confidence=0.9,
        provenance=Provenance(source="claude_code", session_id="s1"),
    )


def _settings(**graph_overrides: object) -> Settings:
    s = Settings()
    return s.model_copy(update={"graph": GraphSettings(**graph_overrides)})  # type: ignore[arg-type]


async def _seed(storage: InMemoryStorage, mems: list[MemoryUnit]) -> None:
    # Distinct entity ids = lower(canonical) so persist_mentions resolves names.
    names: set[str] = set()
    for m in mems:
        names.update(m.entities)
    for name in names:
        await storage.upsert_entity(Entity(id=f"e-{name}", canonical_name=name))
    for m in mems:
        await storage.upsert_memory(m)
    await storage.persist_mentions()


# --- protocol + pure helpers ----------------------------------------------


def test_satisfies_maintenance_job_protocol() -> None:
    job = CoMentionJob(InMemoryStorage())
    assert isinstance(job, MaintenanceJob)
    assert job.name == "co_mention"


def test_co_mention_weight_raw_when_downweight_off() -> None:
    assert co_mention_weight(5, 100, 100, hub_downweight=False) == 5.0


def test_co_mention_weight_downweights_hubs() -> None:
    # A pair sharing two RARE entities (df 2,2) scores far above a pair sharing
    # the same count but via an ultra-frequent hub (df 100, 2).
    rare = co_mention_weight(2, 2, 2, hub_downweight=True)
    hub = co_mention_weight(2, 100, 2, hub_downweight=True)
    assert rare == pytest.approx(2 / math.sqrt(4))
    assert hub == pytest.approx(2 / math.sqrt(200))
    assert rare > hub


# --- the applied pass (run) ------------------------------------------------


@pytest.mark.asyncio
async def test_run_derives_co_mention_edges_from_mentions() -> None:
    storage = InMemoryStorage()
    # m1 mentions rust+tokio; m2 mentions rust+tokio -> shared=2 -> an edge.
    await _seed(
        storage,
        [
            _mem("a", entities=["rust", "tokio"], mid="m1"),
            _mem("b", entities=["rust", "tokio"], mid="m2"),
        ],
    )
    report = await CoMentionJob(storage, settings=_settings()).run()

    assert report.edges_added == 1
    assert ("e-rust", "e-tokio") in storage.co_mentions
    weight, shared = storage.co_mentions[("e-rust", "e-tokio")]
    assert shared == 2
    # Hub down-weight on by default: 2 / sqrt(df_rust * df_tokio) = 2 / sqrt(4).
    assert weight == pytest.approx(2 / math.sqrt(4))


@pytest.mark.asyncio
async def test_min_shared_threshold_filters_weak_pairs() -> None:
    storage = InMemoryStorage()
    # rust+tokio share 1 memory; default min_shared=2 -> no edge.
    await _seed(storage, [_mem("a", entities=["rust", "tokio"], mid="m1")])
    report = await CoMentionJob(storage, settings=_settings()).run()
    assert report.edges_added == 0
    assert storage.co_mentions == {}


@pytest.mark.asyncio
async def test_min_weight_floor_prunes_trivial_links() -> None:
    storage = InMemoryStorage()
    await _seed(
        storage,
        [
            _mem("a", entities=["rust", "tokio"], mid="m1"),
            _mem("b", entities=["rust", "tokio"], mid="m2"),
        ],
    )
    # Weight is 2/sqrt(4) = 1.0; a floor above that prunes the edge.
    report = await CoMentionJob(
        storage, settings=_settings(co_mention_min_weight=1.5)
    ).run()
    assert report.edges_added == 0
    assert storage.co_mentions == {}


@pytest.mark.asyncio
async def test_degree_cap_keeps_highest_weight_edges_per_node() -> None:
    storage = InMemoryStorage()
    # Hub 'core' co-occurs with a, b, c (each shared=2). With max_added_degree=1,
    # 'core' keeps only its single highest-weight edge. Make the weights differ by
    # giving the partners different document-frequencies via extra solo mentions.
    await _seed(
        storage,
        [
            _mem("p1", entities=["core", "a"], mid="m1"),
            _mem("p2", entities=["core", "a"], mid="m2"),
            _mem("q1", entities=["core", "b"], mid="m3"),
            _mem("q2", entities=["core", "b"], mid="m4"),
            _mem("r1", entities=["core", "c"], mid="m5"),
            _mem("r2", entities=["core", "c"], mid="m6"),
            # Bump df(b) and df(c) so their edges weigh LESS than core<->a.
            _mem("solo-b", entities=["b"], mid="m7"),
            _mem("solo-c1", entities=["c"], mid="m8"),
            _mem("solo-c2", entities=["c"], mid="m9"),
        ],
    )
    job = CoMentionJob(storage, settings=_settings(co_mention_max_added_degree=1))
    report = await job.run()

    # 'core' may appear in only ONE surviving edge (its strongest, core<->a, since
    # a has the lowest df). Pairs are stored a<b ordered, so it is ('e-a','e-core').
    core_edges = [
        (a, b) for (a, b) in storage.co_mentions if a == "e-core" or b == "e-core"
    ]
    assert len(core_edges) == 1
    assert ("e-a", "e-core") in storage.co_mentions
    assert report.edges_added == 1


@pytest.mark.asyncio
async def test_run_is_idempotent() -> None:
    storage = InMemoryStorage()
    await _seed(
        storage,
        [
            _mem("a", entities=["rust", "tokio"], mid="m1"),
            _mem("b", entities=["rust", "tokio"], mid="m2"),
        ],
    )
    job = CoMentionJob(storage, settings=_settings())

    first = await job.run()
    after_first = dict(storage.co_mentions)
    second = await job.run()

    # MERGE/upsert (weight re-asserted, not summed): the second pass writes the
    # SAME edges, the store does not grow, and the weights are identical.
    assert first.edges_added == 1
    assert second.edges_added == 1
    assert storage.co_mentions == after_first


@pytest.mark.asyncio
async def test_run_with_no_mentions_is_noop() -> None:
    storage = InMemoryStorage()
    report = await CoMentionJob(storage, settings=_settings()).run()
    assert report.edges_added == 0
    assert report.job_name == "co_mention"


@pytest.mark.asyncio
async def test_merge_entities_repoints_co_mention_edges() -> None:
    # Forward-compat with entity-dedup: when a duplicate entity is merged, the
    # co-mention edges incident to it repoint to the survivor and collapse.
    storage = InMemoryStorage()
    await _seed(
        storage,
        [
            _mem("a", entities=["rust", "tokio"], mid="m1"),
            _mem("b", entities=["rust", "tokio"], mid="m2"),
            _mem("c", entities=["rust-lang", "tokio"], mid="m3"),
            _mem("d", entities=["rust-lang", "tokio"], mid="m4"),
        ],
    )
    await CoMentionJob(storage, settings=_settings()).run()
    assert ("e-rust", "e-tokio") in storage.co_mentions
    assert ("e-rust-lang", "e-tokio") in storage.co_mentions

    await storage.merge_entities("e-rust-lang", "e-rust")

    # The rust-lang<->tokio edge repoints onto rust<->tokio (collapsed, no dup).
    keys = set(storage.co_mentions)
    assert ("e-rust-lang", "e-tokio") not in keys
    assert ("e-rust", "e-tokio") in keys
