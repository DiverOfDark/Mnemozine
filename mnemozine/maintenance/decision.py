"""FR-MNT-1 — the 4-way write decision (add / reinforce / supersede / no-op).

On every write, a new :class:`~mnemozine.schema.models.MemoryUnit` is compared
**only** against existing memories in the *same scope with overlapping
entities* (never a graph-wide scan, FR-RET-2 discipline) and resolved into
exactly one of:

* **add**       — no related memory exists -> insert.
* **reinforce** — a semantically equivalent memory exists -> bump its
  confidence, refresh its timestamp, **no new node**.
* **supersede** — a related global-decision (``ScopeDecision.GLOBAL``) memory
  *contradicts* the new one (e.g. "prefers ``anyhow``" -> "prefers ``thiserror``")
  -> **close the old memory's validity window** (``valid_to = now``, demoting it
  off the hot path) and insert the new one as active. This is the mechanism that
  delivers UC-2 / Goal 2; the temporal model alone does not detect a *reversal*.
* **no-op**     — the new memory is strictly weaker/older than what already
  exists.

Contradiction detection is deliberately cheap and narrow (PRD FR-MNT-1): a
**single** LLM call, fed at most
:attr:`MaintenanceSettings.contradiction_candidate_cap` candidates, restricted
to global-decision (``ScopeDecision.GLOBAL``) units in the same scope sharing >=1
entity. It is never a graph-wide scan.

This module depends only on the :class:`~mnemozine.interfaces.StorageBackend`,
:class:`~mnemozine.interfaces.LLMProvider`, and
:class:`~mnemozine.interfaces.EmbeddingProvider` Protocols, so a concrete
``StorageBackend`` (Graphiti/FalkorDB) can delegate its
:meth:`StorageBackend.upsert_memory` 4-way decision to a :class:`WriteDecider`,
and the decision is unit-testable offline against the conftest fakes.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from mnemozine.activity import emit, write_decision_event
from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import (
    ActivityLog,
    EmbeddingProvider,
    LLMProvider,
    StorageBackend,
    WriteDecision,
    WriteResult,
)
from mnemozine.schema.models import MemoryUnit, ScopeDecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration snapshot for the decision (pulled from Settings, FR-MNT-1/§6.6)
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class WriteDecisionConfig:
    """Tunables governing the FR-MNT-1 write decision (§6.6 "config, not constants").

    Sourced from :class:`~mnemozine.config.MaintenanceSettings` so nothing here is
    a magic number:

    * :attr:`equivalence_threshold` — cosine-similarity cutoff above which a write
      *reinforces* an existing unit instead of *adding* a new node
      (``dedup.equivalence_threshold``).
    * :attr:`contradiction_candidate_cap` — max global-decision candidates fed
      to the single cheap contradiction LLM call
      (``maintenance.contradiction_candidate_cap``).
    """

    equivalence_threshold: float = 0.9
    contradiction_candidate_cap: int = 5

    @classmethod
    def from_settings(cls, settings: Settings) -> WriteDecisionConfig:
        m = settings.maintenance
        return cls(
            equivalence_threshold=m.dedup_equivalence_threshold,
            contradiction_candidate_cap=m.contradiction_candidate_cap,
        )


# ---------------------------------------------------------------------------
# Pure helpers (offline-testable, no I/O)
# ---------------------------------------------------------------------------


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors; ``0.0`` for a zero/empty vector.

    Used to gate reinforce-vs-add against ``dedup.equivalence_threshold``.
    """

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


CONTRADICTION_SYSTEM = (
    "You are a strict preference-reversal detector for a personal memory system. "
    "You are given an operator's NEW durable preference and one EXISTING durable "
    "preference on an overlapping topic. Decide whether the NEW statement REVERSES "
    "or otherwise directly contradicts the EXISTING one such that the EXISTING one "
    "is no longer true (e.g. 'prefers anyhow' vs 'prefers thiserror' for the same "
    "concern). Refinement, addition, or an unrelated preference is NOT a "
    "contradiction. Respond ONLY with strict JSON of the form "
    '{"contradicts": true|false, "reason": "<short>"}.'
)

# Backward-compatible private alias (pre-existing internal name). Prefer the
# public ``CONTRADICTION_SYSTEM`` — it is the stable cross-layer contract reused
# by :mod:`mnemozine.services`.
_CONTRADICTION_SYSTEM = CONTRADICTION_SYSTEM

# JSON schema handed to the LLM provider for the narrow contradiction decision.
CONTRADICTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "contradicts": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["contradicts"],
    "additionalProperties": True,
}


def build_contradiction_prompt(new: MemoryUnit, existing: MemoryUnit) -> str:
    """Render the single-pair contradiction prompt (one LLM call per write)."""

    shared = sorted(set(new.entities) & set(existing.entities))
    return (
        "NEW preference:\n"
        f"  {new.content.strip()}\n"
        "EXISTING preference:\n"
        f"  {existing.content.strip()}\n"
        f"Shared entities: {', '.join(shared) or '(none)'}\n"
        f"Scope: {new.scope.as_str()}\n"
        "Does the NEW preference reverse/contradict the EXISTING one?"
    )


def parse_contradiction(raw: object) -> bool:
    """Coerce an LLM JSON response into a bool, defaulting to ``False`` (no supersede).

    Defaulting to ``False`` is the safe direction: a missed contradiction leaves
    both units active (the older one decays naturally), whereas a false positive
    would wrongly close a still-valid preference window.

    Public, stable API: reused across the storage contradiction predicate
    (:func:`mnemozine.services.make_contradiction_fn`) and the maintenance
    :class:`WriteDecider` so the parse rule lives in one place.
    """

    if isinstance(raw, dict):
        val = raw.get("contradicts")
    else:
        val = raw
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "yes", "1"}
    return False


# Backward-compatible private alias for the pre-existing internal name.
_parse_contradiction = parse_contradiction


# ---------------------------------------------------------------------------
# The decider
# ---------------------------------------------------------------------------


class WriteDecider:
    """Computes the FR-MNT-1 4-way write decision for a single ``upsert``.

    Depends only on the :class:`~mnemozine.interfaces.StorageBackend`,
    :class:`~mnemozine.interfaces.LLMProvider`, and
    :class:`~mnemozine.interfaces.EmbeddingProvider` Protocols. A concrete storage
    backend can delegate its :meth:`StorageBackend.upsert_memory` body to
    :meth:`decide`; this keeps the cheap-narrow-LLM contradiction logic in one
    tested place rather than re-implemented per backend.

    Candidate gathering is strictly bounded (FR-MNT-1): same scope + overlapping
    entities, taken from :meth:`StorageBackend.iter_memories` filtered to the
    write's scope and active-only.
    """

    def __init__(
        self,
        storage: StorageBackend,
        llm: LLMProvider,
        *,
        embeddings: EmbeddingProvider | None = None,
        config: WriteDecisionConfig | None = None,
        settings: Settings | None = None,
        activity_log: ActivityLog | None = None,
    ) -> None:
        self._storage = storage
        self._llm = llm
        self._embeddings = embeddings
        if config is None:
            config = WriteDecisionConfig.from_settings(settings or get_settings())
        self._config = config
        # Optional WEBUI Q3 observability seam: the 4-way write decision is the
        # extract_decision the Logs feed surfaces. Defaults to None so every
        # existing caller is unaffected; emit() fast-paths None / NullActivityLog.
        self._activity_log = activity_log

    # --- candidate gathering (same scope + overlapping entities) ----------

    async def _candidates(self, memory: MemoryUnit) -> list[MemoryUnit]:
        """Active units in the SAME scope sharing >=1 entity (FR-MNT-1 bound)."""

        new_entities = set(memory.entities)
        out: list[MemoryUnit] = []
        async for existing in self._storage.iter_memories(
            scope=memory.scope, active_only=True
        ):
            if existing.id == memory.id:
                continue
            if new_entities & set(existing.entities):
                out.append(existing)
        return out

    # --- contradiction check (cheap, narrow, single LLM call) -------------

    async def _contradicts(
        self, new: MemoryUnit, candidates: Sequence[MemoryUnit]
    ) -> MemoryUnit | None:
        """Run the narrow contradiction check; return the first contradicted unit.

        Only global-decision candidates are considered (FR-MNT-1: the
        MemoryUnit-level *preference-reversal* decision sits on top of Graphiti's
        entity-edge invalidation). In the category-split contract a "preference" is
        a global-scope memory (``ScopeDecision.GLOBAL``), so both the new unit and
        the candidates are gated on that. At most
        :attr:`WriteDecisionConfig.contradiction_candidate_cap` candidates are
        examined, and each is a single cheap LLM call — never a graph-wide scan.
        """

        if new.scope_decision is not ScopeDecision.GLOBAL:
            return None
        pref_candidates = [
            c for c in candidates if c.scope_decision is ScopeDecision.GLOBAL
        ]
        if not pref_candidates:
            return None
        capped = pref_candidates[: self._config.contradiction_candidate_cap]
        for existing in capped:
            prompt = build_contradiction_prompt(new, existing)
            try:
                raw = await self._llm.complete_json(
                    prompt, schema=CONTRADICTION_SCHEMA, system=CONTRADICTION_SYSTEM
                )
            except Exception:  # noqa: BLE001 - never let a flaky LLM block a write
                logger.warning(
                    "contradiction check failed for memory %s vs %s; treating as no-contradiction",
                    new.id,
                    existing.id,
                    exc_info=True,
                )
                continue
            if parse_contradiction(raw):
                return existing
        return None

    # --- equivalence (reinforce vs add) -----------------------------------

    async def _equivalent(
        self, new: MemoryUnit, candidates: Sequence[MemoryUnit]
    ) -> MemoryUnit | None:
        """Find a semantically-equivalent active candidate (reinforce target).

        Exact normalized-content match always counts as equivalent. If an
        :class:`EmbeddingProvider` is available, a cosine similarity at/above
        :attr:`WriteDecisionConfig.equivalence_threshold` also counts
        (``dedup.equivalence_threshold``). Same-category candidates only.
        """

        new_norm = new.content.strip().lower()
        same_type = [c for c in candidates if c.category == new.category]
        # Exact-content equivalence first (cheap, no embedding needed).
        for existing in same_type:
            if existing.content.strip().lower() == new_norm:
                return existing
        if self._embeddings is None:
            return None
        new_vec = await self._embeddings.embed(new.content)
        best: MemoryUnit | None = None
        best_sim = self._config.equivalence_threshold
        for existing in same_type:
            sim = cosine_similarity(
                new_vec, await self._embeddings.embed(existing.content)
            )
            if sim >= best_sim:
                best_sim = sim
                best = existing
        return best

    # --- the decision -----------------------------------------------------

    async def decide(self, memory: MemoryUnit) -> WriteResult:
        """Resolve and apply the 4-way write decision for ``memory``.

        Ordering matters: **reinforce** (equivalent) is checked before
        **supersede** (contradicts) so that a re-statement of the *same*
        preference reinforces rather than closing its own window, and
        **no-op** is the strictly-weaker-duplicate fallback before **add**.

        Side effects are applied through the storage backend Protocol only:
        reinforce mutates the existing unit's confidence/timestamp in place;
        supersede calls :meth:`StorageBackend.close_validity_window` on every
        contradicted unit and inserts the new one; add inserts the new one.
        """

        candidates = await self._candidates(memory)
        result = await self._decide_inner(memory, candidates)
        self._emit_decision(memory, result)
        return result

    async def _decide_inner(
        self, memory: MemoryUnit, candidates: list[MemoryUnit]
    ) -> WriteResult:
        """Resolve + apply the 4-way decision (the side-effecting branches)."""

        # 1) reinforce — a semantically equivalent active memory exists.
        equivalent = await self._equivalent(memory, candidates)
        if equivalent is not None:
            equivalent.confidence = max(equivalent.confidence, memory.confidence)
            equivalent.last_accessed = datetime.now(UTC)
            await self._reinforce_in_store(equivalent)
            return WriteResult(decision=WriteDecision.REINFORCE, memory=equivalent)

        # 2) supersede — a contradicting type=preference memory exists.
        contradicted = await self._contradicts(memory, candidates)
        if contradicted is not None:
            closed = await self._storage.close_validity_window(contradicted.id)
            await self._insert(memory)
            return WriteResult(
                decision=WriteDecision.SUPERSEDE,
                memory=memory,
                superseded=[closed],
            )

        # 3) no-op — strictly weaker/older duplicate-ish memory already present.
        weaker_target = self._strictly_weaker(memory, candidates)
        if weaker_target is not None:
            return WriteResult(decision=WriteDecision.NO_OP, memory=weaker_target)

        # 4) add.
        await self._insert(memory)
        return WriteResult(decision=WriteDecision.ADD, memory=memory)

    def _emit_decision(self, memory: MemoryUnit, result: WriteResult) -> None:
        """Record the FR-MNT-1 4-way write decision on the activity feed (WEBUI Q3).

        Null-safe + error-swallowing (:func:`emit`); a no-op unless an activity log
        is wired through the constructor, so the existing write path is unchanged.
        """

        superseded_ids = [m.id for m in result.superseded]
        emit(
            self._activity_log,
            write_decision_event(
                decision=result.decision.value,
                memory_id=result.memory.id,
                source=memory.provenance.source if memory.provenance else None,
                summary=(
                    f"write {result.decision.value}: {memory.content[:60]}"
                ),
                superseded_ids=superseded_ids,
                detail={
                    "category": memory.category,
                    "cross_ref_candidate": memory.cross_ref_candidate,
                    "scope_decision": memory.scope_decision.value,
                    "scope": memory.scope.as_str(),
                    "entities": list(memory.entities),
                    "confidence": memory.confidence,
                },
            ),
        )

    @staticmethod
    def _strictly_weaker(
        new: MemoryUnit, candidates: Sequence[MemoryUnit]
    ) -> MemoryUnit | None:
        """Return an existing unit the new one is strictly weaker/older than.

        A near-duplicate (same category, same normalized content) that already
        exists with >= the new confidence and a newer-or-equal ``valid_from`` makes
        the write a no-op (nothing to learn).
        """

        new_norm = new.content.strip().lower()
        for existing in candidates:
            if (
                existing.category == new.category
                and existing.content.strip().lower() == new_norm
                and new.confidence < existing.confidence
            ):
                return existing
        return None

    # --- store mutation seams (kept tiny so backends can override) --------

    async def _insert(self, memory: MemoryUnit) -> None:
        """Persist a freshly-decided-active unit.

        Delegates to the backend's enumeration-backed store. The conftest
        ``InMemoryStorage`` exposes its ``memories`` dict; a real backend would
        run its own node write. We avoid calling ``upsert_memory`` here to prevent
        infinite recursion when a backend delegates *to* this decider.
        """

        store_memories = getattr(self._storage, "memories", None)
        if isinstance(store_memories, dict):
            store_memories[memory.id] = memory
            return
        # Fallback for backends that expose a low-level insert hook.
        insert = getattr(self._storage, "insert_memory_node", None)
        if callable(insert):
            await insert(memory)
            return
        raise NotImplementedError(
            "WriteDecider requires the StorageBackend to expose a `memories` dict "
            "or an async `insert_memory_node(memory)` hook for the add/supersede "
            "insert; a Graphiti backend should pass its node-write callable in."
        )

    async def _reinforce_in_store(self, memory: MemoryUnit) -> None:
        """Refresh a reinforced unit's bookkeeping in the backend (best-effort)."""

        store_memories = getattr(self._storage, "memories", None)
        if isinstance(store_memories, dict):
            store_memories[memory.id] = memory


def decision_to_json(result: WriteResult) -> str:
    """Compact JSON summary of a :class:`WriteResult` for audit logs (R5)."""

    return json.dumps(
        {
            "decision": result.decision.value,
            "memory_id": result.memory.id,
            "superseded": [m.id for m in result.superseded],
        },
        sort_keys=True,
    )
