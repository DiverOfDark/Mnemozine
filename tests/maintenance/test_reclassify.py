"""Reclassify + re-extract tests — offline re-application of the current classifier.

Covers the ``ReclassifyJob`` re-tag loop (re-scope / re-categorize stored
memories from their content + provenance, no raw text) and the ``ReExtractJob``
wrapper over the raw-tier re-extraction seam, all offline against a scriptable
fake :class:`~mnemozine.interfaces.Extractor` + the conftest fakes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from mnemozine.config import Settings
from mnemozine.interfaces import (
    Classification,
    MaintenanceJob,
    RetrievalContext,
)
from mnemozine.maintenance.reclassify import ReclassifyJob, ReExtractJob
from mnemozine.schema.events import IngestEvent
from mnemozine.schema.models import (
    MemoryUnit,
    Provenance,
    RawChunk,
    Scope,
    ScopeDecision,
)
from tests.conftest import InMemoryStorage

# ---------------------------------------------------------------------------
# Fake Extractor (scriptable single-statement classifier)
# ---------------------------------------------------------------------------


class FakeExtractor:
    """A scriptable :class:`~mnemozine.interfaces.Extractor` for the reclassify loop.

    ``classify`` is routed by a ``(statement, context) -> Classification``
    callable; ``extract`` is unused by the reclassify path (the raw re-extraction
    is exercised through the storage fake's seam) and records its calls.
    """

    def __init__(
        self,
        classifier: Callable[[str, RetrievalContext], Classification],
    ) -> None:
        self._classifier = classifier
        self.classify_calls: list[str] = []
        self.extract_calls: list[Sequence[IngestEvent]] = []

    async def extract(self, chunk: Sequence[IngestEvent]) -> list[MemoryUnit]:
        self.extract_calls.append(chunk)
        return []

    async def classify(
        self, statement: str, context: RetrievalContext
    ) -> Classification:
        self.classify_calls.append(statement)
        return self._classifier(statement, context)


def _classification(
    *,
    scope: Scope,
    category: str = "fact",
    cross_ref_candidate: bool = False,
) -> Classification:
    return Classification(
        scope_decision=(
            ScopeDecision.GLOBAL if scope.is_global else ScopeDecision.PROJECT
        ),
        scope=scope,
        category=category,
        cross_ref_candidate=cross_ref_candidate,
    )


def _mem(
    content: str,
    *,
    scope: Scope | None = None,
    category: str = "fact",
    cross_ref_candidate: bool = False,
    entities: list[str] | None = None,
) -> MemoryUnit:
    return MemoryUnit(
        content=content,
        scope=scope or Scope.global_(),
        category=category,
        cross_ref_candidate=cross_ref_candidate,
        entities=entities or ["rust"],
        confidence=0.9,
        provenance=Provenance(source="claude_code", session_id="s1"),
    )


# ---------------------------------------------------------------------------
# ReclassifyJob — the re-tag loop
# ---------------------------------------------------------------------------


def test_reclassify_satisfies_protocol() -> None:
    extractor = FakeExtractor(lambda s, c: _classification(scope=Scope.global_()))
    job = ReclassifyJob(InMemoryStorage(), extractor)
    assert isinstance(job, MaintenanceJob)
    assert job.name == "reclassify"


@pytest.mark.asyncio
async def test_reclassify_relabels_changed_category() -> None:
    storage = InMemoryStorage()
    mem = _mem("Prefers thiserror.", category="pref")
    await storage.upsert_memory(mem)
    # Current classifier now emits "preference" for this statement.
    extractor = FakeExtractor(
        lambda s, c: _classification(scope=Scope.global_(), category="preference")
    )

    job = ReclassifyJob(storage, extractor, settings=Settings())
    report = await job.run()

    assert report.consolidated == 1  # one memory re-tagged
    assert storage.memories[mem.id].category == "preference"


@pytest.mark.asyncio
async def test_reclassify_rescopes_memory() -> None:
    storage = InMemoryStorage()
    # Stored as global, but the current classifier scopes it to a project.
    mem = _mem("This project pins tokio 1.38.", scope=Scope.global_())
    await storage.upsert_memory(mem)
    target = Scope.project("Mnemozine")
    extractor = FakeExtractor(lambda s, c: _classification(scope=target))

    job = ReclassifyJob(storage, extractor)
    report = await job.run()

    assert report.consolidated == 1
    assert storage.memories[mem.id].scope.as_str() == target.as_str()


@pytest.mark.asyncio
async def test_reclassify_toggles_cross_ref_flag() -> None:
    storage = InMemoryStorage()
    mem = _mem("Idea: a memory CLI.", cross_ref_candidate=False)
    await storage.upsert_memory(mem)
    extractor = FakeExtractor(
        lambda s, c: _classification(scope=Scope.global_(), cross_ref_candidate=True)
    )

    job = ReclassifyJob(storage, extractor)
    await job.run()

    assert storage.memories[mem.id].cross_ref_candidate is True


@pytest.mark.asyncio
async def test_reclassify_is_idempotent_noop_when_unchanged() -> None:
    storage = InMemoryStorage()
    mem = _mem("Prefers thiserror.", category="preference")
    await storage.upsert_memory(mem)
    # Classifier returns exactly the stored tags -> no change, no write.
    extractor = FakeExtractor(
        lambda s, c: _classification(scope=Scope.global_(), category="preference")
    )

    job = ReclassifyJob(storage, extractor)
    first = await job.run()
    second = await job.run()

    assert first.consolidated == 0  # nothing drifted
    assert second.consolidated == 0


@pytest.mark.asyncio
async def test_reclassify_passes_stored_context_to_classifier() -> None:
    # The loop must classify from STORED content + provenance: it feeds the
    # memory's own content and a context derived from its scope/entities.
    storage = InMemoryStorage()
    mem = _mem(
        "Uses async tokio.",
        scope=Scope.project("Mnemozine"),
        entities=["tokio", "async"],
    )
    await storage.upsert_memory(mem)

    seen: dict[str, object] = {}

    def classifier(statement: str, context: RetrievalContext) -> Classification:
        seen["statement"] = statement
        seen["project"] = context.project
        seen["entities"] = list(context.entities)
        # Ancestor composition: a project scope's context spans global+project.
        seen["scope_strs"] = [s.as_str() for s in context.scopes]
        return _classification(scope=mem.scope, category=mem.category)

    extractor = FakeExtractor(classifier)
    await ReclassifyJob(storage, extractor).run()

    assert seen["statement"] == "Uses async tokio."
    assert seen["project"] == "Mnemozine"
    assert set(seen["entities"]) == {"tokio", "async"}  # type: ignore[arg-type]
    assert "global" in seen["scope_strs"]  # type: ignore[operator]
    assert "project:Mnemozine" in seen["scope_strs"]  # type: ignore[operator]


@pytest.mark.asyncio
async def test_reclassify_classifier_failure_leaves_memory_untouched() -> None:
    storage = InMemoryStorage()
    mem = _mem("Prefers thiserror.", category="pref")
    await storage.upsert_memory(mem)

    def boom(statement: str, context: RetrievalContext) -> Classification:
        raise RuntimeError("classifier down")

    job = ReclassifyJob(storage, FakeExtractor(boom))
    report = await job.run()

    # A flaky classifier must not corrupt the stored tag.
    assert report.consolidated == 0
    assert storage.memories[mem.id].category == "pref"


@pytest.mark.asyncio
async def test_reclassify_scope_filter_restricts_the_sweep() -> None:
    storage = InMemoryStorage()
    g = _mem("global thing", scope=Scope.global_())
    p = _mem("project thing", scope=Scope.project("Mnemozine"))
    await storage.upsert_memory(g)
    await storage.upsert_memory(p)
    extractor = FakeExtractor(
        lambda s, c: _classification(scope=c.scopes[-1], category="changed")
    )

    # Restrict to the project scope only: the global memory must be left alone.
    job = ReclassifyJob(storage, extractor, scope=Scope.project("Mnemozine"))
    await job.run()

    assert storage.memories[p.id].category == "changed"
    assert storage.memories[g.id].category == "fact"
    assert extractor.classify_calls == ["project thing"]


# ---------------------------------------------------------------------------
# ReExtractJob — the raw-tier re-extraction wrapper
# ---------------------------------------------------------------------------


def test_re_extract_satisfies_protocol() -> None:
    extractor = FakeExtractor(lambda s, c: _classification(scope=Scope.global_()))
    job = ReExtractJob(InMemoryStorage(), extractor)
    assert isinstance(job, MaintenanceJob)
    assert job.name == "re_extract"


@pytest.mark.asyncio
async def test_re_extract_supersedes_prior_memories() -> None:
    storage = InMemoryStorage()
    # A prior memory produced by a stored raw chunk.
    old = _mem("old extraction", scope=Scope.project("Mnemozine"))
    await storage.upsert_memory(old)
    chunk = RawChunk(
        content_hash="h1",
        content="some normalized transcript text",
        source="claude_code",
        session_id="s1",
        scope=Scope.project("Mnemozine"),
        project="Mnemozine",
        memory_ids=[old.id],
    )
    await storage.persist_raw_chunk(chunk)
    extractor = FakeExtractor(lambda s, c: _classification(scope=Scope.global_()))

    job = ReExtractJob(storage, extractor, supersede_existing=True)
    report = await job.run()

    assert report.job_name == "re_extract"
    assert report.re_extracted == 1
    # The chunk's prior memory had its validity window closed (replaced).
    assert storage.memories[old.id].valid_to is not None


@pytest.mark.asyncio
async def test_re_extract_keep_existing_does_not_supersede() -> None:
    storage = InMemoryStorage()
    old = _mem("old extraction", scope=Scope.project("Mnemozine"))
    await storage.upsert_memory(old)
    chunk = RawChunk(
        content_hash="h2",
        content="text",
        source="claude_code",
        session_id="s1",
        scope=Scope.project("Mnemozine"),
        project="Mnemozine",
        memory_ids=[old.id],
    )
    await storage.persist_raw_chunk(chunk)
    extractor = FakeExtractor(lambda s, c: _classification(scope=Scope.global_()))

    job = ReExtractJob(storage, extractor, supersede_existing=False)
    report = await job.run()

    assert report.re_extracted == 1
    # supersede_existing=False -> the prior memory stays active.
    assert storage.memories[old.id].is_active


@pytest.mark.asyncio
async def test_re_extract_scope_filter_is_exact() -> None:
    storage = InMemoryStorage()
    a = RawChunk(
        content_hash="ha",
        content="a",
        source="claude_code",
        session_id="sa",
        scope=Scope.project("A"),
        project="A",
    )
    b = RawChunk(
        content_hash="hb",
        content="b",
        source="claude_code",
        session_id="sb",
        scope=Scope.project("B"),
        project="B",
    )
    await storage.persist_raw_chunk(a)
    await storage.persist_raw_chunk(b)
    extractor = FakeExtractor(lambda s, c: _classification(scope=Scope.global_()))

    # Only project:A chunks re-extracted (exact scope, no ancestor composition).
    job = ReExtractJob(storage, extractor, scope=Scope.project("A"))
    report = await job.run()

    assert report.re_extracted == 1
