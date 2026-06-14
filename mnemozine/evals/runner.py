"""Orchestration: load the gold set, (optionally) inflate, run every §9 metric.

:class:`EvalRunner` ties the pieces together:

1. Materialize the gold set's seed memories into a ``StorageBackend`` (the fake
   ``InMemoryStorage`` offline, or a real Graphiti/FalkorDB backend in prod).
2. Optionally inflate the store with synthetic distractors at a given multiplier
   (1x / 10x / 100x) via :class:`~mnemozine.evals.distractors.DistractorGenerator`.
3. Run each §9 metric runner and collect a :class:`~mnemozine.evals.metrics.MetricResult`.

The headline §9 assertion — *precision stays flat as the store grows 10x/100x* —
is :meth:`precision_scaling`: it runs the injection-precision metric at each
inflation level against a freshly-seeded store and asserts no decline.

Components (Retriever / CrossReferencer / Extractor) are injected so production
wires the real implementations; when omitted, the runner falls back to the
self-contained :mod:`mnemozine.evals.harness_adapters` so the whole thing runs
offline against the conftest fakes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from mnemozine.config import Settings
from mnemozine.evals import metrics
from mnemozine.evals.distractors import (
    DEFAULT_INFLATION_LEVELS,
    DistractorGenerator,
)
from mnemozine.evals.goldset import GoldSet, load_gold_set
from mnemozine.evals.harness_adapters import (
    GraphCrossReferencer,
    KeywordExtractor,
    StorageBackedRetriever,
)
from mnemozine.evals.metrics import MetricResult
from mnemozine.interfaces import (
    CrossReferencer,
    EmbeddingProvider,
    Extractor,
    LLMProvider,
    Retriever,
    StorageBackend,
)

# A factory that builds a fresh, empty StorageBackend (so each inflation level
# starts clean). Offline this is ``InMemoryStorage``; prod injects a real one.
StorageFactory = Callable[[], StorageBackend]


@dataclass(slots=True)
class EvalReport:
    """All metric results from one harness run, plus pass/fail rollup."""

    results: list[MetricResult] = field(default_factory=list)
    inflation_multiplier: int = 1

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def by_name(self, name: str) -> MetricResult | None:
        for r in self.results:
            if r.name == name:
                return r
        return None

    def render(self) -> str:
        lines = [f"=== Mnemozine eval report (inflation {self.inflation_multiplier}x) ==="]
        for r in self.results:
            lines.append(r.summary())
            for c in r.cases:
                if not c.passed:
                    lines.append(f"    FAIL case {c.case_id}: {c.detail}")
        lines.append(f"--- overall: {'PASS' if self.passed else 'FAIL'} ---")
        return "\n".join(lines)


@dataclass(slots=True)
class ScalingReport:
    """Precision-at-each-inflation-level report (the headline §9 assertion)."""

    levels: list[int] = field(default_factory=list)
    precisions: list[float] = field(default_factory=list)
    baseline: float = 0.0
    tolerance: float = 0.0
    passed: bool = True

    def render(self) -> str:
        lines = ["=== Precision-stays-flat scaling (PRD §9) ==="]
        for lvl, p in zip(self.levels, self.precisions, strict=False):
            lines.append(f"  {lvl:>4}x  precision={p:.3f}")
        verdict = "PASS" if self.passed else "FAIL"
        lines.append(
            f"--- baseline={self.baseline:.3f} tolerance={self.tolerance:.3f} -> {verdict} ---"
        )
        return "\n".join(lines)


class EvalRunner:
    """Loads the gold set into a store and runs the §9 metrics (offline-capable)."""

    def __init__(
        self,
        *,
        storage_factory: StorageFactory,
        gold_set: GoldSet | None = None,
        settings: Settings | None = None,
        embeddings: EmbeddingProvider | None = None,
        llm: LLMProvider | None = None,
        retriever_factory: Callable[[StorageBackend], Retriever] | None = None,
        crossref_factory: Callable[[StorageBackend], CrossReferencer] | None = None,
        extractor: Extractor | None = None,
        distractor_seed: int = 1234,
    ) -> None:
        self.storage_factory = storage_factory
        self.gold_set = gold_set or load_gold_set()
        self.settings = settings or Settings()
        self.embeddings = embeddings
        self.llm = llm
        self._retriever_factory = retriever_factory or StorageBackedRetriever
        self._crossref_factory = crossref_factory or GraphCrossReferencer
        self._extractor = extractor or KeywordExtractor()
        self.distractor_seed = distractor_seed

    # --- store seeding ---------------------------------------------------

    async def seed_store(self, storage: StorageBackend, *, inflation_multiplier: int = 1) -> int:
        """Load gold memories into ``storage`` and optionally inflate. Returns distractor count.

        Gold memories are inserted first (so they exist regardless of how many
        distractors land), then ``inflation_multiplier`` distractors per gold
        memory are added. ``inflation_multiplier=1`` is the baseline (no extra
        distractors beyond the 1x set); ``0`` seeds gold only.
        """

        for unit in self.gold_set.materialize_memories():
            await storage.upsert_memory(unit)
        if inflation_multiplier <= 0:
            return 0
        gen = DistractorGenerator(self.gold_set, llm=self.llm, seed=self.distractor_seed)
        return await gen.inflate_store(storage, multiplier=inflation_multiplier)

    # --- full metric run -------------------------------------------------

    async def run_all(self, *, inflation_multiplier: int = 1) -> EvalReport:
        """Seed a fresh store and run every §9 metric once."""

        storage = self.storage_factory()
        await self.seed_store(storage, inflation_multiplier=inflation_multiplier)
        retriever = self._retriever_factory(storage)
        crossref = self._crossref_factory(storage)

        results: list[MetricResult] = []
        results.append(await metrics.injection_precision(retriever, self.gold_set))
        results.append(await metrics.changed_preference_correctness(retriever, self.gold_set))
        results.append(await metrics.crossref_precision(crossref, self.gold_set))
        results.append(await metrics.classifier_accuracy(self._extractor, self.gold_set))
        results.append(
            await metrics.retrieval_latency(retriever, self.gold_set, settings=self.settings)
        )
        results.append(await metrics.no_leak_check(storage, self.gold_set))
        await storage.close()
        return EvalReport(results=results, inflation_multiplier=inflation_multiplier)

    # --- precision-stays-flat scaling (the headline §9 metric) -----------

    async def precision_scaling(
        self,
        *,
        levels: tuple[int, ...] | list[int] = DEFAULT_INFLATION_LEVELS,
        tolerance: float = 0.0,
    ) -> ScalingReport:
        """Run injection precision at each inflation level; assert no decline.

        For each level the store is freshly seeded with the gold set + that many
        distractors per gold memory, then injection precision@k is measured. The
        baseline is the precision at the smallest level. The run passes when no
        level's precision drops more than ``tolerance`` below the baseline — i.e.
        precision stays flat as the store grows 10x / 100x (PRD §9 headline).
        """

        ordered = sorted(set(levels))
        precisions: list[float] = []
        for lvl in ordered:
            storage = self.storage_factory()
            await self.seed_store(storage, inflation_multiplier=lvl)
            retriever = self._retriever_factory(storage)
            result = await metrics.injection_precision(retriever, self.gold_set)
            precisions.append(result.score)
            await storage.close()

        baseline = precisions[0] if precisions else 1.0
        passed = all(p >= baseline - tolerance for p in precisions)
        return ScalingReport(
            levels=ordered,
            precisions=precisions,
            baseline=baseline,
            tolerance=tolerance,
            passed=passed,
        )


def default_inmemory_runner(
    *,
    gold_set: GoldSet | None = None,
    settings: Settings | None = None,
) -> EvalRunner:
    """Build a fully-offline runner backed by the packaged ``OfflineStorage``.

    This is the zero-dependency entry the ``mnemozine-eval`` CLI uses: it wires
    the packaged in-memory fake store (so it works post-``pip install`` without
    the test package) plus the harness adapters for
    Retriever/CrossReferencer/Extractor. Runs with no FalkorDB/Ollama/Qwen. The
    unit tests build an :class:`EvalRunner` directly against the conftest
    ``InMemoryStorage`` fake instead.
    """

    from mnemozine.evals._offline_store import OfflineStorage

    return EvalRunner(
        storage_factory=OfflineStorage,
        gold_set=gold_set,
        settings=settings,
    )
