"""Offline (no FalkorDB/Ollama) data-versioning tests against the in-memory fake.

These pin the data-versioning CONTRACT (mnemozine.migrations) on the packaged
in-memory ``StorageBackend`` fake (:class:`mnemozine.evals._offline_store.OfflineStorage`)
so the version persistence + query seam can be validated with no live store:

* a memory / raw-chunk write stamps ``data_version`` (default CURRENT) and reads
  it back; a missing/None stored value coalesces to 0 (legacy/unstamped);
* :meth:`min_data_version` mins over BOTH tiers, is 0 if any record is unstamped,
  and is CURRENT for an empty store;
* :meth:`iter_memories_below_version` / :meth:`iter_chunks_below_version` select
  exactly the records under the target version (legacy coalesced to 0), and the
  :meth:`set_data_version` / :meth:`set_chunk_data_version` stamps are idempotent;
* :meth:`reclassify_memory` and :meth:`re_extract_from_raw_chunks` re-stamp the
  records they touch up to CURRENT (the implicit-stamp migration paths).

The same behaviour is asserted on the conftest ``InMemoryStorage`` fake so the two
in-memory fakes stay behaviourally consistent with each other and with the
FalkorDB backend's contract test.
"""

from __future__ import annotations

import pytest

from mnemozine.evals._offline_store import OfflineStorage
from mnemozine.migrations import (
    CURRENT_DATA_VERSION,
    UNSTAMPED_DATA_VERSION,
    record_data_version,
)
from mnemozine.schema.models import MemoryUnit, Provenance, RawChunk, Scope
from tests.conftest import InMemoryStorage

# Both in-memory StorageBackend fakes must be behaviourally consistent on the
# data-versioning contract; parametrize every test over both.
FAKES = [OfflineStorage, InMemoryStorage]


def _memory(
    *,
    content: str = "a fact",
    scope: Scope | None = None,
    entities: list[str] | None = None,
    mid: str | None = None,
    data_version: int | None = None,
) -> MemoryUnit:
    kwargs: dict = {
        "content": content,
        "scope": scope or Scope.global_(),
        "entities": entities if entities is not None else ["rust"],
        "provenance": Provenance(source="claude_code", session_id="sess-1"),
    }
    if mid is not None:
        kwargs["id"] = mid
    if data_version is not None:
        kwargs["data_version"] = data_version
    return MemoryUnit(**kwargs)


def _raw_chunk(
    *,
    content_hash: str = "h1",
    scope: Scope | None = None,
    memory_ids: list[str] | None = None,
    data_version: int | None = None,
) -> RawChunk:
    sc = scope or Scope.project("Mnemozine")
    kwargs: dict = {
        "content_hash": content_hash,
        "content": "normalized chunk text",
        "source": "claude_code",
        "session_id": "sess-1",
        "scope": sc,
        "project": sc.project_id or "",
        "memory_ids": memory_ids if memory_ids is not None else [],
    }
    if data_version is not None:
        kwargs["data_version"] = data_version
    return RawChunk(**kwargs)


# ---------------------------------------------------------------------------
# record_data_version normalization (the legacy/null -> 0 rule)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0),
        ("not-an-int", 0),
        (object(), 0),
        (0, 0),
        (1, 1),
        (3, 3),
        ("2", 2),
    ],
)
def test_record_data_version_normalizes(value: object, expected: int) -> None:
    assert record_data_version(value) == expected


# ---------------------------------------------------------------------------
# Version persistence (write -> read back) + legacy coalescing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Fake", FAKES)
async def test_memory_write_persists_default_current_version(Fake) -> None:
    store = Fake()
    m = _memory()
    await store.upsert_memory(m)
    assert store.memories[m.id].data_version == CURRENT_DATA_VERSION


@pytest.mark.parametrize("Fake", FAKES)
async def test_raw_chunk_write_persists_default_current_version(Fake) -> None:
    store = Fake()
    await store.persist_raw_chunk(_raw_chunk(content_hash="hc"))
    assert store.raw_chunks["hc"].data_version == CURRENT_DATA_VERSION


@pytest.mark.parametrize("Fake", FAKES)
async def test_explicit_legacy_version_is_preserved_and_coalesced(Fake) -> None:
    store = Fake()
    m = _memory(data_version=UNSTAMPED_DATA_VERSION)
    await store.upsert_memory(m)
    assert record_data_version(store.memories[m.id].data_version) == 0


# ---------------------------------------------------------------------------
# min_data_version: empty store, both tiers, unstamped -> 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Fake", FAKES)
async def test_min_data_version_empty_store_is_current(Fake) -> None:
    store = Fake()
    assert await store.min_data_version() == CURRENT_DATA_VERSION


@pytest.mark.parametrize("Fake", FAKES)
async def test_min_data_version_mins_over_both_tiers(Fake) -> None:
    store = Fake()
    await store.upsert_memory(_memory())
    await store.persist_raw_chunk(_raw_chunk(content_hash="c1"))
    assert await store.min_data_version() == CURRENT_DATA_VERSION

    # An unstamped MEMORY pulls the whole-store min to 0.
    await store.upsert_memory(_memory(mid="legacy", data_version=0, entities=["go"]))
    assert await store.min_data_version() == 0


@pytest.mark.parametrize("Fake", FAKES)
async def test_min_data_version_legacy_chunk_pulls_to_zero(Fake) -> None:
    store = Fake()
    await store.upsert_memory(_memory())
    await store.persist_raw_chunk(_raw_chunk(content_hash="legacy", data_version=0))
    assert await store.min_data_version() == 0


# ---------------------------------------------------------------------------
# iter_*_below_version + set_*_data_version (the migration selection/stamp seam)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Fake", FAKES)
async def test_iter_memories_below_version_selects_only_stale(Fake) -> None:
    store = Fake()
    await store.upsert_memory(_memory(mid="stale", data_version=0))
    await store.upsert_memory(_memory(mid="current", entities=["go"]))

    below = {m.id async for m in store.iter_memories_below_version(CURRENT_DATA_VERSION)}
    assert below == {"stale"}
    # Nothing is below 0.
    assert [m async for m in store.iter_memories_below_version(0)] == []


@pytest.mark.parametrize("Fake", FAKES)
async def test_set_data_version_stamps_counts_and_is_idempotent(Fake) -> None:
    store = Fake()
    await store.upsert_memory(_memory(mid="a", data_version=0))
    await store.upsert_memory(_memory(mid="b", data_version=0, entities=["go"]))

    n = await store.set_data_version(["a", "b"], CURRENT_DATA_VERSION)
    assert n == 2
    assert store.memories["a"].data_version == CURRENT_DATA_VERSION
    assert store.memories["b"].data_version == CURRENT_DATA_VERSION
    # No record remains below the target -> idempotent re-run finds nothing.
    assert [m async for m in store.iter_memories_below_version(CURRENT_DATA_VERSION)] == []
    # Unknown ids / empty list are no-ops.
    assert await store.set_data_version(["missing"], CURRENT_DATA_VERSION) == 0
    assert await store.set_data_version([], CURRENT_DATA_VERSION) == 0


@pytest.mark.parametrize("Fake", FAKES)
async def test_iter_chunks_below_version_and_set_chunk_data_version(Fake) -> None:
    store = Fake()
    await store.persist_raw_chunk(_raw_chunk(content_hash="stale", data_version=0))
    await store.persist_raw_chunk(_raw_chunk(content_hash="fresh"))

    below = {c.content_hash async for c in store.iter_chunks_below_version(CURRENT_DATA_VERSION)}
    assert below == {"stale"}

    n = await store.set_chunk_data_version(["stale"], CURRENT_DATA_VERSION)
    assert n == 1
    assert store.raw_chunks["stale"].data_version == CURRENT_DATA_VERSION
    # Both tiers now reach CURRENT, so min_data_version reaches the target.
    assert [c async for c in store.iter_chunks_below_version(CURRENT_DATA_VERSION)] == []
    assert await store.min_data_version() == CURRENT_DATA_VERSION
    assert await store.set_chunk_data_version([], CURRENT_DATA_VERSION) == 0


# ---------------------------------------------------------------------------
# reclassify / re-extract re-stamp the touched record (implicit migration paths)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Fake", FAKES)
async def test_reclassify_memory_bumps_data_version(Fake) -> None:
    store = Fake()
    await store.upsert_memory(_memory(mid="rc", data_version=0))
    assert store.memories["rc"].data_version == 0

    # A re-tag bumps the version...
    updated = await store.reclassify_memory("rc", category="decision")
    assert updated.data_version == CURRENT_DATA_VERSION
    assert store.memories["rc"].data_version == CURRENT_DATA_VERSION

    # ...and so does a no-field reclassify (always re-stamps).
    store.memories["rc"].data_version = 0
    again = await store.reclassify_memory("rc")
    assert again.data_version == CURRENT_DATA_VERSION


@pytest.mark.parametrize("Fake", FAKES)
async def test_re_extract_from_raw_chunks_bumps_chunk_version(Fake) -> None:
    store = Fake()
    await store.persist_raw_chunk(
        _raw_chunk(content_hash="rx", scope=Scope.project("Mnemozine"), data_version=0)
    )

    class _NoopExtractor:
        async def extract(self, chunk):  # type: ignore[no-untyped-def]
            return []

        async def classify(self, statement, context):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    report = await store.re_extract_from_raw_chunks(
        _NoopExtractor(),  # type: ignore[arg-type]
        scope=Scope.project("Mnemozine"),
    )
    assert report.re_extracted == 1
    assert store.raw_chunks["rx"].data_version == CURRENT_DATA_VERSION
