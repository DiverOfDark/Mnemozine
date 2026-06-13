"""Eval-set bootstrap: auto-propose candidates, operator labels yes/no (PRD §9).

PRD §9 "Eval-set construction (bootstrap — USER-TASK DEPENDENCY)": during the
Phase-1 historical backlog import, the pipeline auto-proposes extracted
candidates and the **operator** labels them yes/no in a quick CLI/markdown review
pass (~40 cases, ≈2–3 hrs). This is an *operator deliverable*, not a Claude Code
deliverable — so this module builds the machinery (propose, render a review
sheet, parse the operator's labels back) but the actual yes/no judgement is the
human's.

Flow:

1. :func:`propose_candidates` runs the extractor over backfilled ingest chunks
   (FR-ING-6 unit) and turns each extracted :class:`MemoryUnit` into a
   :class:`Candidate` with a stable id + provenance.
2. :func:`render_review_markdown` writes a Markdown review sheet: one block per
   candidate with a ``- [ ] keep`` checkbox the operator ticks, plus the
   proposed type/scope/entities to confirm or correct.
3. The operator edits the file (ticking keeps, optionally correcting the type).
4. :func:`parse_review_markdown` reads the edited sheet back into labeled
   :class:`Candidate`s, and :func:`candidates_to_gold_set` folds the kept ones
   into a :class:`~mnemozine.evals.goldset.GoldSet` (seed memories + classifier
   cases) ready to commit and run.

Everything is offline-capable: with the conftest ``FakeLLMProvider`` driving a
fake ``Extractor``, ``propose_candidates`` produces deterministic candidates with
no live model.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from mnemozine.evals.goldset import (
    ClassifierCase,
    GoldMemory,
    GoldSet,
)
from mnemozine.interfaces import Extractor
from mnemozine.schema.events import IngestEvent
from mnemozine.schema.models import MemoryType, MemoryUnit, Scope

# Label values an operator can assign in the review pass.
LABEL_UNREVIEWED = "unreviewed"
LABEL_KEEP = "keep"
LABEL_DROP = "drop"


class Candidate(BaseModel):
    """One auto-proposed eval candidate awaiting the operator's yes/no (PRD §9).

    Carries the proposed memory plus a stable ``candidate_id`` so the rendered
    review sheet and the parsed-back labels line up. ``label`` starts
    ``unreviewed``; the operator sets it to ``keep`` / ``drop``. ``corrected_type``
    lets the operator fix a mis-classification during review (R1 human-in-the-loop).
    """

    candidate_id: str
    content: str
    proposed_type: MemoryType
    scope: str
    entities: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    source_session: str = ""
    label: str = LABEL_UNREVIEWED
    corrected_type: MemoryType | None = None

    @property
    def final_type(self) -> MemoryType:
        """The operator-corrected type if any, else the proposed type."""

        return self.corrected_type or self.proposed_type

    @property
    def kept(self) -> bool:
        return self.label == LABEL_KEEP

    @classmethod
    def from_memory(cls, unit: MemoryUnit, *, index: int) -> Candidate:
        """Build a candidate from an extracted :class:`MemoryUnit`."""

        sess = unit.provenance.session_id if unit.provenance else ""
        return cls(
            candidate_id=f"cand-{index:04d}",
            content=unit.content,
            proposed_type=unit.type,
            scope=unit.scope.as_str(),
            entities=list(unit.entities),
            confidence=unit.confidence,
            source_session=sess,
        )


async def propose_candidates(
    extractor: Extractor,
    chunks: Sequence[Sequence[IngestEvent]] | AsyncIterator[Sequence[IngestEvent]],
) -> list[Candidate]:
    """Extract candidates from backfilled chunks for operator review (PRD §9).

    Runs the (real or fake) ``Extractor`` over each backlog chunk and flattens
    the extracted memory units into review candidates with stable ids. Accepts
    either a concrete sequence of chunks or an async iterator of them (so it can
    be driven straight off ``IngestSource.backfill`` grouped into chunks). The
    operator's yes/no is applied later via the review sheet — this only proposes.
    """

    candidates: list[Candidate] = []
    index = 0

    async def _consume(chunk: Sequence[IngestEvent]) -> None:
        nonlocal index
        units = await extractor.extract(chunk)
        for unit in units:
            candidates.append(Candidate.from_memory(unit, index=index))
            index += 1

    if isinstance(chunks, AsyncIterator):
        async for chunk in chunks:
            await _consume(chunk)
    else:
        for chunk in chunks:
            await _consume(chunk)
    return candidates


# ---------------------------------------------------------------------------
# Markdown review sheet (the operator-facing artifact)
# ---------------------------------------------------------------------------

_HEADER = """\
# Mnemozine eval-set bootstrap — operator review (PRD §9)

Tick `- [x] keep` for each candidate that belongs in the gold eval set, leave it
unticked to drop. Optionally fix the `type:` line (preference | project_fact |
idea_seed) if the classifier got it wrong. Target ~40 high-quality cases.

Save the file, then run `mnemozine-eval bootstrap-finish` to fold the kept
candidates into the committed gold set.
"""

_BLOCK_RE = re.compile(
    r"^## (?P<cid>\S+)\n"
    r"- \[(?P<check>[ xX])\] keep\n"
    r"> (?P<content>.*)\n"
    r"type: (?P<type>\S+)\n"
    r"scope: (?P<scope>\S+)\n"
    r"entities: (?P<entities>.*)\n"
    r"session: (?P<session>.*)$",
    re.MULTILINE,
)


def render_review_markdown(candidates: Sequence[Candidate]) -> str:
    """Render candidates as a Markdown review sheet the operator edits (PRD §9)."""

    blocks: list[str] = [_HEADER]
    for c in candidates:
        check = "x" if c.kept else " "
        ents = ", ".join(c.entities)
        blocks.append(
            f"## {c.candidate_id}\n"
            f"- [{check}] keep\n"
            f"> {c.content}\n"
            f"type: {c.final_type.value}\n"
            f"scope: {c.scope}\n"
            f"entities: {ents}\n"
            f"session: {c.source_session}\n"
        )
    return "\n".join(blocks)


def write_review_markdown(candidates: Sequence[Candidate], path: str | Path) -> Path:
    """Write the review sheet to disk for the operator to edit."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_review_markdown(candidates), encoding="utf-8")
    return out


def parse_review_markdown(text: str) -> list[Candidate]:
    """Parse an operator-edited review sheet back into labeled candidates.

    A ticked ``- [x] keep`` becomes ``label='keep'``; unticked becomes
    ``label='drop'``. The ``type:`` line is read back as the (possibly corrected)
    final type — stored as ``corrected_type`` so a correction is explicit.
    """

    out: list[Candidate] = []
    for m in _BLOCK_RE.finditer(text):
        checked = m.group("check").lower() == "x"
        try:
            final_type = MemoryType(m.group("type").strip())
        except ValueError:
            final_type = MemoryType.PREFERENCE
        entities = [e.strip() for e in m.group("entities").split(",") if e.strip()]
        out.append(
            Candidate(
                candidate_id=m.group("cid"),
                content=m.group("content").strip(),
                proposed_type=final_type,
                corrected_type=final_type,
                scope=m.group("scope").strip(),
                entities=entities,
                source_session=m.group("session").strip(),
                label=LABEL_KEEP if checked else LABEL_DROP,
            )
        )
    return out


def read_review_markdown(path: str | Path) -> list[Candidate]:
    """Read + parse an operator-edited review sheet from disk."""

    return parse_review_markdown(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Folding kept candidates into a gold set
# ---------------------------------------------------------------------------


def candidates_to_gold_set(
    candidates: Sequence[Candidate],
    *,
    name: str = "mnemozine-gold",
    description: str = "Operator-labeled gold set (PRD §9 bootstrap).",
) -> GoldSet:
    """Fold operator-kept candidates into a :class:`GoldSet`.

    Each kept candidate becomes (a) a seed :class:`GoldMemory` and (b) a
    :class:`ClassifierCase` (statement = content, expected type = the operator's
    final type) so the classifier-accuracy metric (R1) runs against real
    operator-labeled data. Dropped/unreviewed candidates are skipped. Other
    metric cases (injection/preference/crossref/no-leak) are left empty for the
    operator to add by hand or via a richer review — this gives a runnable,
    classifier-graded gold set immediately.
    """

    kept = [c for c in candidates if c.kept]
    memories: list[GoldMemory] = []
    classifier_cases: list[ClassifierCase] = []
    for c in kept:
        scope = Scope.parse(c.scope)
        memories.append(
            GoldMemory(
                gold_id=c.candidate_id,
                type=c.final_type,
                content=c.content,
                scope=c.scope,
                entities=list(c.entities),
                confidence=c.confidence,
            )
        )
        classifier_cases.append(
            ClassifierCase(
                case_id=f"cls-{c.candidate_id}",
                statement=c.content,
                project=scope.project_id,
                expected_type=c.final_type,
            )
        )
    return GoldSet(
        name=name,
        description=description,
        memories=memories,
        classifier_cases=classifier_cases,
    )


def review_stats(candidates: Sequence[Candidate]) -> dict[str, int]:
    """Counts for a quick CLI summary of review progress (PRD §9 ~40 cases)."""

    kept = sum(1 for c in candidates if c.label == LABEL_KEEP)
    dropped = sum(1 for c in candidates if c.label == LABEL_DROP)
    unreviewed = sum(1 for c in candidates if c.label == LABEL_UNREVIEWED)
    return {
        "total": len(candidates),
        "keep": kept,
        "drop": dropped,
        "unreviewed": unreviewed,
    }
