"""FR-MNT-2 — tiered consolidation: raw transcript -> extracted fact -> theme.

Retrieval operates on the *distilled* tiers, not raw transcripts. This periodic
pass merges related facts (same scope, overlapping entities, high semantic
similarity) into a single higher-level consolidated unit so the hot store stays
compact as it grows — the PRD's "consolidate rather than accumulate" constraint.

Mechanics, all through the :class:`~mnemozine.interfaces.StorageBackend` and
:class:`~mnemozine.interfaces.LLMProvider` Protocols:

1. Enumerate active hot units per scope (:meth:`StorageBackend.iter_memories`).
2. Cluster by overlapping entities + embedding similarity
   (``dedup.equivalence_threshold`` reused as the merge cutoff so the knob stays
   in one place).
3. For each cluster of >= 2 units, ask the LLM for one consolidated statement,
   insert it as a fresh active unit, and **archive** (never delete) the source
   units it subsumes.

Idempotent (FR-MNT-5): a freshly-written consolidated unit is single-membered on
the next run (its sources are archived and excluded), so re-running consolidates
nothing already consolidated. A degenerate/empty LLM response leaves the cluster
untouched rather than dropping memories.
"""

from __future__ import annotations

import logging

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import (
    EmbeddingProvider,
    LLMProvider,
    MaintenanceReport,
    StorageBackend,
)
from mnemozine.maintenance.decision import cosine_similarity
from mnemozine.schema.models import MemoryUnit, Provenance, Scope, Tier

logger = logging.getLogger(__name__)

_CONSOLIDATE_SYSTEM = (
    "You consolidate a small cluster of an operator's related memory statements "
    "into ONE concise higher-level statement that preserves every distinct fact. "
    "Do not invent facts; do not drop any. Respond with the consolidated "
    "statement as plain text only."
)


def build_consolidation_prompt(cluster: list[MemoryUnit]) -> str:
    """Render the cluster -> single-theme consolidation prompt."""

    lines = "\n".join(f"- {m.content.strip()}" for m in cluster)
    return f"Consolidate these related statements into one:\n{lines}"


class ConsolidationJob:
    """FR-MNT-2 tiered consolidation pass.

    Requires an :class:`~mnemozine.interfaces.EmbeddingProvider` to cluster and an
    :class:`~mnemozine.interfaces.LLMProvider` to synthesize the theme. Depends on
    nothing but the Protocols.
    """

    def __init__(
        self,
        storage: StorageBackend,
        llm: LLMProvider,
        embeddings: EmbeddingProvider,
        *,
        settings: Settings | None = None,
    ) -> None:
        self._storage = storage
        self._llm = llm
        self._embeddings = embeddings
        self._settings = settings or get_settings()

    @property
    def name(self) -> str:
        return "consolidation"

    async def run(self) -> MaintenanceReport:
        report = MaintenanceReport(job_name=self.name)
        # Group active hot units by scope so we never cross scope boundaries
        # (a project_fact must not consolidate into a global preference).
        by_scope: dict[str, list[MemoryUnit]] = {}
        async for mem in self._storage.iter_memories(active_only=True, tier=Tier.HOT):
            by_scope.setdefault(mem.scope.as_str(), []).append(mem)

        threshold = self._settings.maintenance.dedup_equivalence_threshold
        for scope_str, units in by_scope.items():
            clusters = await self._cluster(units, threshold)
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                consolidated = await self._consolidate_cluster(
                    Scope.parse(scope_str), cluster
                )
                if consolidated is None:
                    continue
                # Archive (retain, never delete) the subsumed source units.
                for src in cluster:
                    await self._storage.archive(src.id)
                report.consolidated += 1
                report.notes.append(
                    f"consolidated {len(cluster)} unit(s) in scope '{scope_str}' "
                    f"-> {consolidated.id}"
                )
        return report

    # --- clustering (entity overlap + embedding similarity) ---------------

    async def _cluster(
        self, units: list[MemoryUnit], threshold: float
    ) -> list[list[MemoryUnit]]:
        """Greedy single-link clustering by entity overlap AND embedding similarity.

        Only units of the same free-form ``category`` and sharing >=1 entity are
        candidates; a cosine similarity at/above ``threshold`` joins them. Greedy
        and deterministic (input order), which is enough for the periodic merge.
        """

        # Precompute embeddings once per unit.
        vectors: dict[str, list[float]] = {}
        for u in units:
            vectors[u.id] = await self._embeddings.embed(u.content)

        clusters: list[list[MemoryUnit]] = []
        assigned: set[str] = set()
        for i, u in enumerate(units):
            if u.id in assigned:
                continue
            cluster = [u]
            assigned.add(u.id)
            for v in units[i + 1 :]:
                if v.id in assigned:
                    continue
                if v.category != u.category:
                    continue
                if not (set(u.entities) & set(v.entities)):
                    continue
                if cosine_similarity(vectors[u.id], vectors[v.id]) >= threshold:
                    cluster.append(v)
                    assigned.add(v.id)
            clusters.append(cluster)
        return clusters

    # --- synthesis --------------------------------------------------------

    async def _consolidate_cluster(
        self, scope: Scope, cluster: list[MemoryUnit]
    ) -> MemoryUnit | None:
        """Ask the LLM for one consolidated statement and insert it as active.

        Returns the new unit, or ``None`` if the LLM yields nothing usable (in
        which case the cluster is left untouched — no memory is lost).
        """

        prompt = build_consolidation_prompt(cluster)
        try:
            text = await self._llm.complete(prompt, system=_CONSOLIDATE_SYSTEM)
        except Exception:  # noqa: BLE001 - a flaky LLM must not destroy memories
            logger.warning("consolidation LLM call failed; leaving cluster intact", exc_info=True)
            return None
        text = (text or "").strip()
        if not text:
            return None

        # Union of entities; highest source confidence; merged provenance. The
        # cluster is single-category (see _cluster), so the consolidated theme
        # carries that category; it is a cross-ref seed iff any member was.
        entities = sorted({e for m in cluster for e in m.entities})
        confidence = max(m.confidence for m in cluster)
        category = cluster[0].category
        cross_ref_candidate = any(m.cross_ref_candidate for m in cluster)
        consolidated = MemoryUnit(
            content=text,
            scope=scope,
            category=category,
            cross_ref_candidate=cross_ref_candidate,
            entities=entities,
            confidence=confidence,
            provenance=self._merged_provenance(cluster),
            tier=Tier.HOT,
        )
        result = await self._storage.upsert_memory(consolidated)
        return result.memory

    @staticmethod
    def _merged_provenance(cluster: list[MemoryUnit]) -> Provenance:
        """Carry the first source's provenance forward (theme keeps a real link)."""

        first = cluster[0].provenance
        return Provenance(
            source=first.source,
            session_id=first.session_id,
            chunk_hash=first.chunk_hash,
            raw_path=first.raw_path,
        )
