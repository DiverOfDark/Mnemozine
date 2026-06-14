"""The fixed gold-set data model and loader (PRD §9 eval-set construction).

The gold set encodes *the operator's* preferences across *their* projects, so
only the operator can label it (PRD §9 "USER-TASK DEPENDENCY"). This module
defines the durable, storage-agnostic shape of a labeled eval set and a loader
for the committed JSON fixture, so the whole harness can run **offline against
fakes** without a live store.

A :class:`GoldSet` carries the inputs every §9 metric needs:

* **memories** — the seed MemoryUnits to load into the store under test (the
  "gold" memories whose surfacing the precision metric scores).
* **injection cases** — for FR-RET-3/5 injection precision@k: a query/context
  with the set of memory ids that *should* surface and that *should not*.
* **preference cases** — for the changed-preference metric (UC-2): which memory
  is the current value vs the stale value that must not surface as active.
* **crossref cases** — for FR-RET-6 cross-reference precision: the connections a
  context should surface, judged relevant/irrelevant.
* **classifier cases** — for the make-or-break R1 classifier-accuracy metric: a
  bare statement plus its gold ``global``/``project`` scope-decision label.
* **no-leak cases** — for the no-leak check: a ``project_fact`` and the unrelated
  project scope it must never appear in.

The JSON fixture (``fixtures/gold_set.json``) is intentionally small but covers
every metric so ``mnemozine-eval`` and the unit tests run end-to-end with no
network. A real operator gold set (≈40 cases, PRD §9) has the same shape and is
produced by the bootstrap CLI (:mod:`mnemozine.evals.bootstrap`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from mnemozine.schema.models import (
    DEFAULT_CATEGORY,
    MemoryUnit,
    Provenance,
    Scope,
    ScopeDecision,
    Tier,
)

# Where the committed offline fixture lives.
FIXTURES_DIR = Path(__file__).parent / "fixtures"
DEFAULT_GOLD_SET_PATH = FIXTURES_DIR / "gold_set.json"


def _eval_provenance(case_id: str) -> Provenance:
    """A stable provenance stamp for a gold memory (audit-friendly, R5)."""

    return Provenance(source="eval", session_id=f"gold:{case_id}")


class GoldMemory(BaseModel):
    """A seed memory unit for the eval store, plus a stable gold id.

    ``gold_id`` is a human-readable, fixture-stable identifier (e.g.
    ``pref-rust-errors``) that the metric cases reference. It is distinct from
    the runtime ``MemoryUnit.id`` (a random uuid) so the fixture can wire cases
    to memories without pinning uuids. :meth:`to_memory` materializes the actual
    :class:`MemoryUnit` (with a deterministic id derived from ``gold_id`` so two
    loads of the same fixture produce stable ids).
    """

    gold_id: str = Field(description="Fixture-stable id referenced by metric cases.")
    category: str = Field(
        default=DEFAULT_CATEGORY,
        description=(
            "FREE-FORM emergent classifier category (no fixed enum); replaces the "
            "old MemoryType. The scope decision (global vs project) is implied by "
            "the persisted 'scope' string, so it is not carried separately."
        ),
    )
    cross_ref_candidate: bool = Field(
        default=False,
        description="True for a cross-reference seed (the old idea_seed flag).",
    )
    content: str
    scope: str = Field(description="Persisted scope string: 'global' or 'project:<id>'.")
    entities: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    tier: Tier = Tier.HOT
    # Optional relative age in days (older = larger). Used so the changed-pref
    # case can make the stale value strictly older than the current one without
    # pinning absolute timestamps in the fixture.
    age_days: float = 0.0
    # If set, the memory is loaded with its validity window already closed
    # (superseded) — i.e. a stale preference that must NOT surface as active.
    superseded: bool = False

    def to_memory(self, *, now: datetime | None = None) -> MemoryUnit:
        """Materialize a runtime :class:`MemoryUnit` from this gold memory."""

        now = now or datetime.now(UTC)
        valid_from = now - timedelta(days=self.age_days)
        unit = MemoryUnit(
            id=_gold_memory_id(self.gold_id),
            category=self.category,
            cross_ref_candidate=self.cross_ref_candidate,
            content=self.content,
            scope=Scope.parse(self.scope),
            entities=list(self.entities),
            confidence=self.confidence,
            provenance=_eval_provenance(self.gold_id),
            tier=self.tier,
            valid_from=valid_from,
        )
        if self.superseded:
            # Close the window *after* valid_from so the unit is well-formed.
            unit.supersede(at=valid_from + timedelta(seconds=1))
        return unit


def _gold_memory_id(gold_id: str) -> str:
    """Deterministic runtime memory id derived from a fixture-stable gold id."""

    return f"gold-{gold_id}"


class InjectionCase(BaseModel):
    """A precision@k case for SessionStart/mid-session injection (FR-RET-3/5).

    Given ``query`` + the working ``context`` (project + active entities), the
    retriever produces ranked memories. ``should_surface`` lists the gold ids
    that are correct to surface (relevant); ``should_not_surface`` lists gold ids
    that are wrong to surface (distractors / out-of-scope). Precision@k is the
    fraction of the top-k retrieved that are in ``should_surface``.
    """

    case_id: str
    query: str
    project: str | None = None
    scopes: list[str] = Field(
        default_factory=lambda: ["global"],
        description="Composed scope strings to search.",
    )
    entities: list[str] = Field(default_factory=list)
    should_surface: list[str] = Field(
        default_factory=list, description="Gold ids that SHOULD surface."
    )
    should_not_surface: list[str] = Field(
        default_factory=list, description="Gold ids that should NOT surface."
    )
    top_k: int = 5


class PreferenceCase(BaseModel):
    """A changed-preference correctness case (UC-2, PRD §9 preference correctness).

    The operator changed their mind: ``current_gold_id`` is the value that must be
    returned as active; ``stale_gold_id`` is the superseded value that must NOT
    surface as active. The metric queries and asserts current-in, stale-out.
    """

    case_id: str
    query: str
    project: str | None = None
    scopes: list[str] = Field(default_factory=lambda: ["global"])
    entities: list[str] = Field(default_factory=list)
    current_gold_id: str
    stale_gold_id: str
    top_k: int = 5


class CrossRefCase(BaseModel):
    """A cross-reference precision case (FR-RET-6, PRD §9 cross-reference quality).

    For the working ``context``, the cross-referencer surfaces related idea/seed
    nodes. ``relevant_gold_ids`` are connections judged relevant; any surfaced
    connection not in that set counts against precision (a wrong "this reminds me
    of…" is worse than a miss — precision over recall).
    """

    case_id: str
    project: str | None = None
    entities: list[str] = Field(default_factory=list)
    relevant_gold_ids: list[str] = Field(default_factory=list)
    max_suggestions: int = 2


class ClassifierCase(BaseModel):
    """A classifier-accuracy case: a bare statement + its gold decision (R1, §9).

    The single make-or-break metric. ``statement`` is fed to
    ``Extractor.classify`` and the returned :class:`Classification` is compared
    against the gold labels. Under the category split the *controlled* decision
    that gates everything else is the :class:`ScopeDecision` (``global`` vs
    ``project``); the free-form category is emergent and not graded for exact
    equality (it has no fixed enum). ``expected_cross_ref`` optionally grades the
    cross-reference flag (the old ``idea_seed`` distinction).
    """

    case_id: str
    statement: str
    project: str | None = None
    expected_scope_decision: ScopeDecision
    expected_cross_ref: bool | None = Field(
        default=None,
        description="If set, also grade the cross-reference flag for this case.",
    )


class NoLeakCase(BaseModel):
    """A no-leak case: a project_fact and an unrelated project it must not reach.

    PRD §9 no-leak check: a ``project_fact`` scoped to its own project must never
    appear when querying an *unrelated* project. ``fact_gold_id`` is the
    project_fact; ``unrelated_project`` is the foreign project scope to query.
    """

    case_id: str
    fact_gold_id: str
    unrelated_project: str
    query: str
    entities: list[str] = Field(default_factory=list)
    top_k: int = 10


class GoldSet(BaseModel):
    """A fixed, operator-labeled eval set (PRD §9).

    Carries the seed memories plus per-metric cases. Built either from the
    committed offline fixture (:func:`load_gold_set`) or by the operator review
    pass (:mod:`mnemozine.evals.bootstrap`). Storage-agnostic: the runner loads
    :attr:`memories` into whatever ``StorageBackend`` is under test.
    """

    name: str = "mnemozine-gold"
    description: str = ""
    memories: list[GoldMemory] = Field(default_factory=list)
    injection_cases: list[InjectionCase] = Field(default_factory=list)
    preference_cases: list[PreferenceCase] = Field(default_factory=list)
    crossref_cases: list[CrossRefCase] = Field(default_factory=list)
    classifier_cases: list[ClassifierCase] = Field(default_factory=list)
    no_leak_cases: list[NoLeakCase] = Field(default_factory=list)

    def memory_by_gold_id(self, gold_id: str) -> GoldMemory:
        """Look up a seed memory by its fixture-stable gold id."""

        for m in self.memories:
            if m.gold_id == gold_id:
                return m
        raise KeyError(f"no gold memory with gold_id={gold_id!r}")

    def runtime_id(self, gold_id: str) -> str:
        """The runtime ``MemoryUnit.id`` a given gold id materializes to."""

        return _gold_memory_id(gold_id)

    def materialize_memories(self, *, now: datetime | None = None) -> list[MemoryUnit]:
        """Materialize all seed memories into runtime :class:`MemoryUnit`s."""

        now = now or datetime.now(UTC)
        return [m.to_memory(now=now) for m in self.memories]


def load_gold_set(path: str | Path | None = None) -> GoldSet:
    """Load a :class:`GoldSet` from a JSON file (defaults to the committed fixture).

    With ``path=None`` this loads ``fixtures/gold_set.json``, the small committed
    gold set that lets the whole harness run offline against the fakes in
    ``tests/conftest.py`` (PRD §9: "small committed gold-set fixture so the
    harness runs offline against fakes").
    """

    path = Path(path) if path is not None else DEFAULT_GOLD_SET_PATH
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return GoldSet.model_validate(data)


def save_gold_set(gold_set: GoldSet, path: str | Path) -> Path:
    """Serialize a :class:`GoldSet` to JSON (used by the bootstrap review pass)."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(gold_set.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def all_gold_ids(gold_set: GoldSet) -> set[str]:
    """Every gold id referenced anywhere in the set (for validation/tests)."""

    ids: set[str] = {m.gold_id for m in gold_set.memories}
    return ids


def runtime_ids(gold_set: GoldSet, gold_ids: Iterable[str]) -> set[str]:
    """Map a collection of fixture gold ids to their runtime memory ids."""

    return {gold_set.runtime_id(g) for g in gold_ids}
