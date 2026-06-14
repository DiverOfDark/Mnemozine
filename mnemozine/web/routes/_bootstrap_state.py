"""In-process state for the F4 eval-bootstrap browser labeling flow (PRD §4.8).

The F4 trio — ``GET /api/eval/bootstrap`` (list), ``POST .../{id}/label``, and
``POST .../bootstrap/finish`` — needs a shared, mutable candidate queue so a label
applied by one request is visible to the next and to ``finish``. The eval harness
itself (:mod:`mnemozine.evals.bootstrap`) is *stateless* (it proposes candidates
and folds kept ones into a gold set); persisting *which* candidates the operator
has labeled, across requests within a console session, is a WebUI concern.

This module holds that queue as a single process-wide
:class:`BootstrapStore`. It self-seeds with a small deterministic candidate set on
first use (matching the read-route stub's ``cand-0000`` so the label endpoint works
end-to-end against the contract), and ``finish`` folds the kept candidates into a
gold set written to :attr:`BootstrapStore.gold_set_path` (a working file, never the
committed package fixture). The store is in-memory only — labels do not survive a
process restart, which is acceptable for a single-operator local console.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from mnemozine.evals.bootstrap import (
    LABEL_UNREVIEWED,
    Candidate,
)
from mnemozine.schema.models import MemoryType


def _default_candidates() -> list[Candidate]:
    """The deterministic seed queue (offline, no extractor needed).

    Mirrors the read-route's sample candidate (``cand-0000``) so the label/finish
    endpoints operate on the same ids the SPA sees from ``GET /api/eval/bootstrap``,
    plus a couple more so the fold/finish path has something to keep.
    """

    return [
        Candidate(
            candidate_id="cand-0000",
            content="I prefer thiserror over anyhow in Rust.",
            proposed_type=MemoryType.PREFERENCE,
            scope="global",
            entities=["rust", "error-handling"],
            confidence=0.9,
            source_session="backlog-1",
            label=LABEL_UNREVIEWED,
        ),
        Candidate(
            candidate_id="cand-0001",
            content="This project pins tokio 1.38.",
            proposed_type=MemoryType.PROJECT_FACT,
            scope="project:rust-cli",
            entities=["tokio"],
            confidence=0.85,
            source_session="backlog-1",
            label=LABEL_UNREVIEWED,
        ),
        Candidate(
            candidate_id="cand-0002",
            content="Idea: a CLI that diffs two FalkorDB graphs.",
            proposed_type=MemoryType.IDEA_SEED,
            scope="global",
            entities=["falkordb", "cli"],
            confidence=0.7,
            source_session="backlog-2",
            label=LABEL_UNREVIEWED,
        ),
    ]


class BootstrapStore:
    """A process-wide, mutable queue of eval-bootstrap candidates (F4).

    Self-seeds lazily on first access. ``label`` updates one candidate's label /
    corrected type; ``all`` lists the current queue; ``replace`` swaps the whole
    queue (used when a real ``propose_candidates`` run feeds the WebUI).
    """

    def __init__(self) -> None:
        self._candidates: dict[str, Candidate] | None = None
        self._gold_set_path: Path | None = None

    def _ensure_seeded(self) -> dict[str, Candidate]:
        if self._candidates is None:
            self._candidates = {c.candidate_id: c for c in _default_candidates()}
        return self._candidates

    @property
    def gold_set_path(self) -> Path:
        """Working path the folded gold set is written to by ``finish``.

        Defaults to a stable file under the OS temp dir so a real run persists the
        labeled gold set without clobbering the committed package fixture. Tests
        may override it via :meth:`set_gold_set_path`.
        """

        if self._gold_set_path is None:
            self._gold_set_path = Path(tempfile.gettempdir()) / "mnemozine-bootstrap-gold.json"
        return self._gold_set_path

    def set_gold_set_path(self, path: str | Path) -> None:
        """Override where ``finish`` writes the folded gold set (tests / config)."""

        self._gold_set_path = Path(path)

    def all(self) -> list[Candidate]:
        """Return the current candidate queue in stable id order."""

        return list(self._ensure_seeded().values())

    def get(self, candidate_id: str) -> Candidate | None:
        """Return one candidate by id, or ``None`` if unknown."""

        return self._ensure_seeded().get(candidate_id)

    def label(
        self,
        candidate_id: str,
        *,
        label: str,
        corrected_type: MemoryType | None = None,
    ) -> Candidate | None:
        """Apply a label (and optional reclassification) to one candidate.

        Returns the updated candidate, or ``None`` when the id is unknown.
        """

        store = self._ensure_seeded()
        current = store.get(candidate_id)
        if current is None:
            return None
        updated = current.model_copy(
            update={"label": label, "corrected_type": corrected_type}
        )
        store[candidate_id] = updated
        return updated

    def replace(self, candidates: list[Candidate]) -> None:
        """Replace the whole queue (e.g. after a real propose run)."""

        self._candidates = {c.candidate_id: c for c in candidates}

    def reset(self) -> None:
        """Drop all state so the next access re-seeds (test isolation)."""

        self._candidates = None


# The process-wide store shared by the F4 read route and the label/finish routes.
bootstrap_store = BootstrapStore()


__all__ = ["BootstrapStore", "bootstrap_store"]
