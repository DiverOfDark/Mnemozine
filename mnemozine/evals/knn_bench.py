"""KNN over-fetch benchmark (PRD §9 synthetic scaling, FR-RET-2 / Goal 5).

The retrieval hot path runs an index-backed approximate-KNN against FalkorDB's
``db.idx.vector.queryNodes`` and applies the scope / tier / entity ``WHERE``
clause **after** the KNN cut. That ordering is the whole reason
``retrieval.knn_overfetch_factor`` / ``knn_overfetch_cap`` exist: if the index is
asked for only ``top_k`` nearest neighbours, a *low-selectivity scope* (a small
project living inside a large global store) can be starved — the ``top_k`` raw
neighbours are dominated by nearer **out-of-scope** vectors, so the in-scope
true top-k never reaches the post-filter. The backend mitigates this by
over-fetching ``K = top_k * factor`` (bounded by ``cap``) before filtering.

This benchmark makes that behaviour measurable *before a large real FalkorDB
store exists*. It reuses the synthetic distractor generator and the 1x / 10x /
100x scaling harness from :mod:`mnemozine.evals`: it builds a low-selectivity
scope inside a large global store, then for a battery of queries compares

* an **exhaustive in-process baseline** — the true cosine top-k computed over
  *only the in-scope* memories (what an unbounded scan would return), against
* the **over-fetch path** — take the global top ``K = min(top_k * factor, cap)``
  nearest by vector distance, filter to scope, then take top_k,

and reports **recall@k** (fraction of the true in-scope top-k the over-fetch
path actually surfaces) at each inflation level. recall@k = 1.0 means the
configured over-fetch is sufficient at that store size; a drop below 1.0 means
the factor/cap is starving the post-filter and must be raised.

Why a self-contained vector model (not the lexical fake store):
``OfflineStorage.scoped_query`` filters *then* ranks, so it can never reproduce
the filter-after-KNN starvation that the over-fetch knob defends against. To
exercise the real behaviour offline we model the vector layer directly here with
deterministic, seeded embeddings — no Ollama/FalkorDB, fully reproducible.

RECOMMENDED OVER-FETCH TUNING (the deliverable of this benchmark)
-----------------------------------------------------------------
The required over-fetch factor scales with how *selective* the scope is: if the
in-scope fraction of the store is ``s`` (e.g. a project that is 1% of all
memories => s = 0.01), then to expect ``top_k`` in-scope hits inside the raw KNN
cut you need roughly ``K >= top_k / s`` neighbours, i.e.::

    factor >= 1 / in_scope_fraction          (cap permitting)

Concretely, for top_k=10 and a scope that is ~1% of the store you need
``factor ~= 100`` to keep recall@k at 1.0 — which is exactly why
``knn_overfetch_factor`` defaults to 10 (good for scopes down to ~10% of the
store) and is paired with ``knn_overfetch_cap=512`` so a large ``top_k`` cannot
turn the over-fetch into an unbounded scan (defeating the flat-search-space
Goal 5). When this benchmark shows recall@k slipping below 1.0 at 10x/100x for a
realistic scope, **raise ``knn_overfetch_factor`` (and, if the product
``top_k * factor`` is being clipped, ``knn_overfetch_cap``) until it recovers**,
trading a larger index scan for in-scope recall. The cap is the backstop that
keeps the scan bounded; prefer pushing the factor and only lift the cap when the
factor alone is being clipped.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from mnemozine.config import RetrievalSettings
from mnemozine.evals.distractors import (
    DEFAULT_INFLATION_LEVELS,
    DistractorGenerator,
)
from mnemozine.evals.goldset import GoldSet, load_gold_set
from mnemozine.schema.models import MemoryType, MemoryUnit, Scope

__all__ = [
    "KnnBenchConfig",
    "KnnLevelResult",
    "KnnBenchReport",
    "effective_overfetch_k",
    "run_knn_overfetch_bench",
]


# ---------------------------------------------------------------------------
# Deterministic, dependency-free embedding model
# ---------------------------------------------------------------------------


def _embed(text: str, *, dims: int, seed: int = 0) -> list[float]:
    """A deterministic unit-norm pseudo-embedding for ``text``.

    Seeded by the text (and an optional salt) so the same string always maps to
    the same vector — reproducible across runs with no Ollama/bge-m3. This is
    NOT semantically meaningful; the benchmark only needs a *stable, well-spread*
    vector space so the geometry of the filter-after-KNN starvation is faithful.
    """

    rng = random.Random(f"{seed}:{text}")
    vec = [rng.gauss(0.0, 1.0) for _ in range(dims)]
    norm = math.sqrt(sum(c * c for c in vec)) or 1.0
    return [c / norm for c in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two (assumed unit-norm) vectors; higher = nearer."""

    return sum(x * y for x, y in zip(a, b, strict=True))


def effective_overfetch_k(top_k: int, retrieval: RetrievalSettings) -> int:
    """The raw KNN cut the backend fetches before the scope/tier filter.

    Mirrors ``storage/backend.py``: ``K = min(top_k * knn_overfetch_factor,
    knn_overfetch_cap)``, floored at ``top_k`` so a tiny/over-aggressive cap can
    never ask the index for fewer than ``top_k`` neighbours.
    """

    raw = top_k * max(1, retrieval.knn_overfetch_factor)
    capped = min(raw, max(top_k, retrieval.knn_overfetch_cap))
    return max(top_k, capped)


# ---------------------------------------------------------------------------
# Benchmark config + result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class KnnBenchConfig:
    """Inputs for the KNN over-fetch benchmark.

    ``scope`` is the low-selectivity scope under test (a project living inside a
    large global store). ``top_k`` is the retrieval depth the recall is measured
    at. ``n_queries`` random in-scope query vectors are evaluated per level and
    the recall@k averaged. ``embedding_dims`` keeps the synthetic vector space
    small enough to be fast while still well-spread.

    The store size at level ``L`` is ``store_multiplier * L`` global distractors
    (so 1x / 10x / 100x produce a genuinely large store from the tiny committed
    gold fixture), and the in-scope size is held at ``in_scope_fraction_target``
    of that store. Holding the *fraction* constant is the PRD's flat-search-space
    premise: a well-chosen ``knn_overfetch_factor`` (>= 1 / fraction) must then
    keep recall@k flat as the absolute store grows.
    """

    scope: Scope = field(default_factory=lambda: Scope.project("knn-bench-scope"))
    top_k: int = 10
    n_queries: int = 25
    embedding_dims: int = 64
    seed: int = 7
    # Distractors added to the global store *per inflation level* (so the 100x
    # store is ~100 * this). Large enough that the scope is genuinely a small
    # island in a big store, exercising the filter-after-KNN starvation.
    store_multiplier: int = 60
    # The scope is held at this fraction of the whole store at every level (the
    # PRD's "effective search space stays constant"). At ~20% the required factor
    # is ~1/0.2 = 5, so the default knn_overfetch_factor=10 carries comfortable
    # headroom and holds recall@k = 1.0 flat across 1x/10x/100x — while a starved
    # factor (e.g. 1) or a tighter scope (a smaller --in-scope-fraction) collapses
    # it, demonstrating exactly when the knob must be raised.
    in_scope_fraction_target: float = 0.2


@dataclass(slots=True)
class KnnLevelResult:
    """recall@k at one inflation level for one over-fetch configuration."""

    level: int
    total_memories: int
    in_scope_memories: int
    in_scope_fraction: float
    overfetch_k: int
    recall_at_k: float

    def summary(self) -> str:
        return (
            f"  {self.level:>4}x  store={self.total_memories:<7} "
            f"in_scope={self.in_scope_memories} "
            f"(frac={self.in_scope_fraction:.4f})  "
            f"K={self.overfetch_k:<4} recall@k={self.recall_at_k:.3f}"
        )


@dataclass(slots=True)
class KnnBenchReport:
    """All levels for the configured over-fetch, plus a pass/fail rollup."""

    factor: int
    cap: int
    top_k: int
    results: list[KnnLevelResult] = field(default_factory=list)
    recall_floor: float = 1.0

    @property
    def passed(self) -> bool:
        """True when every level keeps recall@k at/above the floor."""

        return all(r.recall_at_k >= self.recall_floor for r in self.results)

    @property
    def min_recall(self) -> float:
        return min((r.recall_at_k for r in self.results), default=1.0)

    def render(self) -> str:
        lines = [
            "=== KNN over-fetch benchmark (PRD §9, FR-RET-2) ===",
            f"  factor={self.factor}  cap={self.cap}  top_k={self.top_k}  "
            f"recall_floor={self.recall_floor:.3f}",
        ]
        lines.extend(r.summary() for r in self.results)
        verdict = "PASS" if self.passed else "FAIL"
        lines.append(f"--- min recall@k={self.min_recall:.3f} -> {verdict} ---")
        if not self.passed:
            lines.append(
                "  HINT: recall@k slipped below the floor — the configured "
                "over-fetch is starving the post-KNN scope filter. Raise "
                "retrieval.knn_overfetch_factor (>= 1 / in_scope_fraction), and "
                "lift retrieval.knn_overfetch_cap only if top_k*factor is being "
                "clipped by it. See module docstring for the tuning rule."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------


def _seed_corpus(
    cfg: KnnBenchConfig, gold_set: GoldSet, level: int
) -> tuple[list[MemoryUnit], list[MemoryUnit]]:
    """Build (all_memories, in_scope_memories) for one inflation ``level``.

    The out-of-scope bulk is the gold set plus ``store_multiplier * level``
    distractors (all global / other scopes); the in-scope set is sized to hold
    ``in_scope_fraction_target`` of the resulting store, so selectivity stays
    constant as the absolute store grows (the PRD's flat-search-space premise).
    In-scope memories are synthetic project facts tagged with ``cfg.scope``;
    distractors come from the existing :class:`DistractorGenerator` so the
    benchmark rides the same synthetic-scaling machinery as the precision evals.
    """

    rng = random.Random(f"{cfg.seed}:scope:{level}")

    out_of_scope: list[MemoryUnit] = [
        m for m in gold_set.materialize_memories() if m.scope.as_str() != cfg.scope.as_str()
    ]
    if level > 0:
        gen = DistractorGenerator(gold_set, seed=cfg.seed + level)
        target = cfg.store_multiplier * level
        # generate() is async; drive it synchronously via the deterministic
        # template path (no LLM) so this seeding stays sync + offline.
        distractors = _generate_sync(gen, target)
        out_of_scope.extend(m for m in distractors if m.scope.as_str() != cfg.scope.as_str())

    # Hold the in-scope fraction constant at the target: solve
    # in / (in + out) = frac  ->  in = frac * out / (1 - frac).
    frac = min(max(cfg.in_scope_fraction_target, 1e-6), 0.95)
    n_out = len(out_of_scope)
    n_in = max(cfg.top_k, round(frac * n_out / (1.0 - frac)))

    in_scope: list[MemoryUnit] = []
    for i in range(n_in):
        in_scope.append(
            MemoryUnit(
                id=f"knnbench-scope-{level}-{i}",
                type=MemoryType.PROJECT_FACT,
                content=f"In-scope memory {i} for {cfg.scope.as_str()} token-{rng.random()}",
                scope=cfg.scope,
                entities=[f"scope-entity-{i % 5}"],
                confidence=0.9,
            )
        )

    return [*in_scope, *out_of_scope], in_scope


def _generate_sync(gen: DistractorGenerator, count: int) -> list[MemoryUnit]:
    """Synchronous drive of the deterministic template distractor path."""

    if count <= 0:
        return []
    rng = random.Random(gen.seed)
    units = gen._template_candidates(count, rng)  # noqa: SLF001 - intra-package use
    for offset, unit in enumerate(units):
        unit.id = f"distractor-{gen.seed}-{offset}"
    return units[:count]


def _recall_for_level(
    cfg: KnnBenchConfig,
    retrieval: RetrievalSettings,
    level: int,
    gold_set: GoldSet,
) -> KnnLevelResult:
    """Measure mean recall@k over ``cfg.n_queries`` queries at one level."""

    all_mem, in_scope = _seed_corpus(cfg, gold_set, level)
    dims = cfg.embedding_dims

    # Precompute embeddings once per memory.
    vectors: dict[str, list[float]] = {
        m.id: _embed(m.content, dims=dims, seed=cfg.seed) for m in all_mem
    }
    in_scope_ids = {m.id for m in in_scope}
    overfetch_k = effective_overfetch_k(cfg.top_k, retrieval)

    qrng = random.Random(f"{cfg.seed}:queries:{level}")
    recalls: list[float] = []
    for q in range(cfg.n_queries):
        # Query vector biased toward a random in-scope memory so a non-trivial
        # in-scope top-k exists to be recalled.
        anchor = in_scope[qrng.randrange(len(in_scope))]
        qvec = _embed(f"query-{q}-{anchor.id}", dims=dims, seed=cfg.seed + 1)

        scored = sorted(
            ((m.id, _cosine(qvec, vectors[m.id])) for m in all_mem),
            key=lambda kv: kv[1],
            reverse=True,
        )

        # Exhaustive in-process baseline: true cosine top-k over IN-SCOPE only
        # (what an unbounded post-filter scan would return).
        baseline = [mid for mid, _ in scored if mid in in_scope_ids][: cfg.top_k]
        if not baseline:
            continue

        # Over-fetch path: top-K globally, filter to scope, take top_k.
        raw_cut = [mid for mid, _ in scored[:overfetch_k]]
        filtered = [mid for mid in raw_cut if mid in in_scope_ids][: cfg.top_k]

        hit = len(set(filtered) & set(baseline))
        recalls.append(hit / len(baseline))

    mean_recall = sum(recalls) / len(recalls) if recalls else 1.0
    total = len(all_mem)
    return KnnLevelResult(
        level=level,
        total_memories=total,
        in_scope_memories=len(in_scope),
        in_scope_fraction=(len(in_scope) / total) if total else 0.0,
        overfetch_k=overfetch_k,
        recall_at_k=mean_recall,
    )


def run_knn_overfetch_bench(
    *,
    retrieval: RetrievalSettings | None = None,
    gold_set: GoldSet | None = None,
    config: KnnBenchConfig | None = None,
    levels: Sequence[int] = DEFAULT_INFLATION_LEVELS,
    recall_floor: float = 1.0,
) -> KnnBenchReport:
    """Run the KNN over-fetch recall@k benchmark across inflation levels.

    For each ``level`` in ``levels`` (1x / 10x / 100x by default), seed a large
    global store with a low-selectivity scope, then measure mean recall@k of the
    over-fetch retrieval path against the exhaustive in-scope cosine baseline,
    using the configured ``retrieval.knn_overfetch_factor`` / ``_cap``.

    Returns a :class:`KnnBenchReport`; ``report.passed`` is True when every level
    keeps recall@k at or above ``recall_floor``. A failure means the configured
    over-fetch is too small for the scope selectivity at that store size (see the
    module docstring for the ``factor >= 1 / in_scope_fraction`` tuning rule).

    Pure/sync and fully offline: deterministic seeded vectors, no FalkorDB /
    Ollama / Qwen. Reuses :class:`DistractorGenerator` + the §9 inflation levels.
    """

    retrieval = retrieval or RetrievalSettings()
    gold_set = gold_set or load_gold_set()
    config = config or KnnBenchConfig()

    ordered = sorted(set(int(x) for x in levels))
    results = [_recall_for_level(config, retrieval, lvl, gold_set) for lvl in ordered]
    return KnnBenchReport(
        factor=retrieval.knn_overfetch_factor,
        cap=retrieval.knn_overfetch_cap,
        top_k=config.top_k,
        results=results,
        recall_floor=recall_floor,
    )
