"""Unit tests for the eval-set bootstrap (PRD §9 USER-TASK DEPENDENCY)."""

from __future__ import annotations

from datetime import UTC, datetime

from mnemozine.evals.bootstrap import (
    LABEL_DROP,
    LABEL_KEEP,
    Candidate,
    candidates_to_gold_set,
    parse_review_markdown,
    propose_candidates,
    render_review_markdown,
    review_stats,
)
from mnemozine.interfaces import Classification, RetrievalContext
from mnemozine.schema.events import IngestEvent, Role, Source
from mnemozine.schema.models import (
    MemoryType,
    MemoryUnit,
    Provenance,
    Scope,
    ScopeDecision,
)


class _ScriptedExtractor:
    """Tiny Extractor returning a fixed unit per chunk (drives propose offline)."""

    def __init__(self, units_per_chunk: list[list[MemoryUnit]]) -> None:
        self._units = units_per_chunk
        self._i = 0

    async def extract(self, chunk):  # noqa: ANN001, ANN201
        out = self._units[self._i] if self._i < len(self._units) else []
        self._i += 1
        return out

    async def classify(self, statement: str, context: RetrievalContext) -> Classification:
        return Classification(
            scope_decision=ScopeDecision.GLOBAL,
            scope=Scope.global_(),
            category="preference",
        )


def _unit(content: str, mtype: MemoryType, scope: Scope) -> MemoryUnit:
    # The bootstrap review sheet still speaks the legacy MemoryType, so the
    # helper takes one and reverse-maps it onto the category-split MemoryUnit.
    return MemoryUnit(
        category=mtype.category,
        cross_ref_candidate=mtype.is_cross_ref,
        content=content,
        scope=scope,
        entities=["rust"],
        confidence=0.8,
        provenance=Provenance(source="claude_code", session_id="s1"),
    )


def _events() -> list[IngestEvent]:
    return [
        IngestEvent(
            source=Source.CLAUDE_CODE,
            project="demo",
            session_id="s1",
            timestamp=datetime(2026, 6, 13, tzinfo=UTC),
            role=Role.USER,
            content="I prefer thiserror.",
        )
    ]


async def test_propose_candidates_from_chunks() -> None:
    extractor = _ScriptedExtractor(
        [
            [
                _unit("prefers thiserror", MemoryType.PREFERENCE, Scope.global_()),
                _unit("pins tokio 1.38", MemoryType.PROJECT_FACT, Scope.project("p")),
            ]
        ]
    )
    cands = await propose_candidates(extractor, [_events()])
    assert len(cands) == 2
    assert cands[0].candidate_id == "cand-0000"
    assert cands[1].candidate_id == "cand-0001"
    assert cands[0].proposed_type is MemoryType.PREFERENCE
    assert cands[1].scope == "project:p"


async def test_propose_candidates_from_async_iterator() -> None:
    extractor = _ScriptedExtractor(
        [[_unit("prefers ruff", MemoryType.PREFERENCE, Scope.global_())]]
    )

    async def _chunks():  # noqa: ANN202
        yield _events()

    cands = await propose_candidates(extractor, _chunks())
    assert len(cands) == 1
    assert cands[0].content == "prefers ruff"


def test_render_then_parse_roundtrip_preserves_fields() -> None:
    cands = [
        Candidate(
            candidate_id="cand-0000",
            content="prefers thiserror over anyhow",
            proposed_type=MemoryType.PREFERENCE,
            scope="global",
            entities=["rust", "errors"],
            source_session="s1",
        ),
        Candidate(
            candidate_id="cand-0001",
            content="project pins tokio 1.38",
            proposed_type=MemoryType.PROJECT_FACT,
            scope="project:rust-cli",
            entities=["tokio"],
            source_session="s2",
        ),
    ]
    md = render_review_markdown(cands)
    parsed = parse_review_markdown(md)
    assert len(parsed) == 2
    assert parsed[0].candidate_id == "cand-0000"
    assert parsed[0].content == "prefers thiserror over anyhow"
    assert parsed[0].entities == ["rust", "errors"]
    assert parsed[1].scope == "project:rust-cli"
    # Nothing ticked -> all dropped.
    assert all(c.label == LABEL_DROP for c in parsed)


def test_operator_tick_marks_keep() -> None:
    cands = [
        Candidate(
            candidate_id="cand-0000",
            content="prefers thiserror",
            proposed_type=MemoryType.PREFERENCE,
            scope="global",
            entities=["rust"],
        )
    ]
    md = render_review_markdown(cands).replace("- [ ] keep", "- [x] keep")
    parsed = parse_review_markdown(md)
    assert parsed[0].label == LABEL_KEEP
    assert parsed[0].kept


def test_operator_type_correction_is_captured() -> None:
    cands = [
        Candidate(
            candidate_id="cand-0000",
            content="this project pins tokio 1.38",
            proposed_type=MemoryType.PREFERENCE,  # wrong proposal
            scope="project:rust-cli",
            entities=["tokio"],
        )
    ]
    md = render_review_markdown(cands)
    # Operator ticks keep AND fixes the type to project_fact.
    md = md.replace("- [ ] keep", "- [x] keep").replace("type: preference", "type: project_fact")
    parsed = parse_review_markdown(md)
    assert parsed[0].final_type is MemoryType.PROJECT_FACT
    assert parsed[0].corrected_type is MemoryType.PROJECT_FACT


def test_candidates_to_gold_set_only_keeps_kept() -> None:
    cands = [
        Candidate(
            candidate_id="c0",
            content="prefers thiserror",
            proposed_type=MemoryType.PREFERENCE,
            scope="global",
            entities=["rust"],
            label=LABEL_KEEP,
        ),
        Candidate(
            candidate_id="c1",
            content="dropped one",
            proposed_type=MemoryType.PREFERENCE,
            scope="global",
            entities=["x"],
            label=LABEL_DROP,
        ),
    ]
    gs = candidates_to_gold_set(cands)
    assert len(gs.memories) == 1
    assert gs.memories[0].gold_id == "c0"
    # Each kept candidate yields a classifier case (R1 metric).
    assert len(gs.classifier_cases) == 1
    assert gs.classifier_cases[0].expected_scope_decision is ScopeDecision.GLOBAL


def test_candidates_to_gold_set_project_scope_sets_case_project() -> None:
    cands = [
        Candidate(
            candidate_id="c0",
            content="pins tokio 1.38",
            proposed_type=MemoryType.PROJECT_FACT,
            scope="project:rust-cli",
            entities=["tokio"],
            label=LABEL_KEEP,
        )
    ]
    gs = candidates_to_gold_set(cands)
    assert gs.classifier_cases[0].project == "rust-cli"
    assert gs.classifier_cases[0].expected_scope_decision is ScopeDecision.PROJECT


def test_review_stats_counts() -> None:
    cands = [
        Candidate(
            candidate_id="c0",
            content="a",
            proposed_type=MemoryType.PREFERENCE,
            scope="global",
            label=LABEL_KEEP,
        ),
        Candidate(
            candidate_id="c1",
            content="b",
            proposed_type=MemoryType.PREFERENCE,
            scope="global",
            label=LABEL_DROP,
        ),
        Candidate(
            candidate_id="c2",
            content="c",
            proposed_type=MemoryType.PREFERENCE,
            scope="global",
        ),
    ]
    stats = review_stats(cands)
    assert stats == {"total": 3, "keep": 1, "drop": 1, "unreviewed": 1}


def test_kept_gold_set_runs_classifier_metric() -> None:
    # End-to-end: a bootstrapped gold set is a runnable gold set.
    cands = [
        Candidate(
            candidate_id="c0",
            content="prefers thiserror",
            proposed_type=MemoryType.PREFERENCE,
            scope="global",
            entities=["rust"],
            label=LABEL_KEEP,
        )
    ]
    gs = candidates_to_gold_set(cands)
    assert gs.classifier_cases
    assert gs.memory_by_gold_id("c0").category == "preference"
