"""Weighted entity-entity co-mention edges with hub down-weight + degree cap.

The :class:`~mnemozine.maintenance.mentions.MentionsJob` turns each memory's
``m.entities`` name list into real ``MNEMOZINE_MENTIONS`` edges. This job derives
the next connectivity layer from those: two entities mentioned by the SAME memory
co-occur, and the more memories they share the stronger the link. Left raw, that
layer collapses into a hairball around a few ultra-frequent hub entities
(``operator`` / ``web`` / ``test`` per the diagnosis), so this job:

1. enumerates the co-occurring entity-id pairs and their shared-memory counts
   (:meth:`StorageBackend.co_mention_pairs`, a pure read-only seam) plus the
   per-entity document-frequency (:meth:`StorageBackend.entity_mention_counts`);
2. computes a TF-IDF-style **down-weighted** weight per pair
   (``shared / sqrt(df_a * df_b)``) when ``graph.co_mention_hub_downweight`` so a
   pair sharing the ubiquitous ``operator`` entity scores far below a pair sharing
   two rare entities, and drops pairs below ``graph.co_mention_min_weight``;
3. enforces a per-node **degree cap** (``graph.co_mention_max_added_degree``):
   each entity keeps only its highest-weight co-mention edges so the added layer
   stays bounded (the hairball cannot form), mirroring
   :meth:`EntityResolutionJob._prune_and_cap`;
4. idempotently MERGEs each surviving pair via
   :meth:`StorageBackend.upsert_co_mention` (distinct ``MNEMOZINE_CO_MENTIONS``
   type, weight re-asserted not summed) so a re-run writes the same edges.

ALL the weighting / cap logic lives HERE (in Python) so it is offline
unit-testable; the backend method is a dumb idempotent upsert. Implements
:class:`~mnemozine.interfaces.MaintenanceJob` and depends only on the
:class:`~mnemozine.interfaces.StorageBackend` Protocol.
"""

from __future__ import annotations

import logging
import math

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import MaintenanceReport, StorageBackend

logger = logging.getLogger(__name__)


def co_mention_weight(
    shared: int, df_a: int, df_b: int, *, hub_downweight: bool
) -> float:
    """Weight of a co-mention pair, optionally hub-down-weighted (pure helper).

    With ``hub_downweight`` the weight is the TF-IDF-style
    ``shared / sqrt(df_a * df_b)`` — the shared-memory count divided by the
    geometric mean of each endpoint's mention document-frequency, so a pair held
    together only by an ultra-frequent hub entity scores far below a pair of two
    rare entities. Without it the weight is the raw ``shared`` count. Kept a free
    function so the policy is deterministic and unit-testable with no storage.
    """

    if not hub_downweight:
        return float(shared)
    denom = math.sqrt(max(df_a, 1) * max(df_b, 1))
    return float(shared) / denom if denom else float(shared)


class CoMentionJob:
    """Derive + upsert the weighted entity-entity co-mention layer (graph connectivity).

    Reads the mention-derived co-occurrence enumeration, applies the hub
    down-weight + per-node degree cap in Python, and idempotently MERGEs the
    surviving co-mention edges. Safe to re-run (FR-MNT-5): the upsert re-asserts
    weight (not sum) and the same pairs survive the deterministic ranking, so a
    second pass writes the same edges and grows nothing.
    """

    def __init__(
        self,
        storage: StorageBackend,
        *,
        settings: Settings | None = None,
    ) -> None:
        self._storage = storage
        self._settings = settings or get_settings()

    @property
    def name(self) -> str:
        return "co_mention"

    async def run(self) -> MaintenanceReport:
        """Compute + upsert the down-weighted, degree-capped co-mention edges.

        Records the number of co-mention edges asserted in
        :attr:`~mnemozine.interfaces.MaintenanceReport.edges_added` plus a note.
        """

        report = MaintenanceReport(job_name=self.name)
        g = self._settings.graph

        pairs = await self._storage.co_mention_pairs(min_shared=g.co_mention_min_shared)
        if not pairs:
            report.notes.append("no co-occurring entity pairs (no co-mention edges)")
            return report
        df = await self._storage.entity_mention_counts()

        kept = self._rank_downweight_and_cap(pairs, df)

        asserted = 0
        for a, b, weight, shared in kept:
            await self._storage.upsert_co_mention(a, b, weight=weight, shared=shared)
            asserted += 1
        report.edges_added = asserted
        report.notes.append(
            f"asserted {asserted} entity-entity co-mention edge(s) "
            f"(MNEMOZINE_CO_MENTIONS, min_shared={g.co_mention_min_shared}, "
            f"hub_downweight={g.co_mention_hub_downweight}, "
            f"max_added_degree={g.co_mention_max_added_degree})"
        )
        return report

    # --- weighting + cap (pure Python; offline-unit-testable) --------------

    def _rank_downweight_and_cap(
        self,
        pairs: list[tuple[str, str, int]],
        df: dict[str, int],
    ) -> list[tuple[str, str, float, int]]:
        """Down-weight, floor-filter, and degree-cap the co-mention pairs.

        Returns the surviving ``(a, b, weight, shared)`` tuples (``a < b``,
        deterministic order) the run should upsert. The per-node degree cap keeps
        each entity's highest-weight edges (ties broken by shared count then the
        pair ids, so it is stable across runs — FR-MNT-5) and drops the overflow,
        bounding the added layer.
        """

        g = self._settings.graph
        min_weight = g.co_mention_min_weight
        max_degree = g.co_mention_max_added_degree

        weighted: list[tuple[str, str, float, int]] = []
        for a, b, shared in pairs:
            weight = co_mention_weight(
                shared,
                df.get(a, 0),
                df.get(b, 0),
                hub_downweight=g.co_mention_hub_downweight,
            )
            if weight < min_weight:
                continue
            weighted.append((a, b, weight, shared))

        # Deterministic global order: highest weight first, then shared, then ids.
        weighted.sort(key=lambda t: (-t[2], -t[3], t[0], t[1]))

        if max_degree < 0:
            return weighted

        # Per-node degree cap: a pair survives only if BOTH endpoints still have
        # room. Walking in descending-weight order keeps each node's strongest
        # edges (EntityResolutionJob._prune_and_cap pattern, extended to a pair).
        degree: dict[str, int] = {}
        kept: list[tuple[str, str, float, int]] = []
        for a, b, weight, shared in weighted:
            if degree.get(a, 0) >= max_degree or degree.get(b, 0) >= max_degree:
                continue
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
            kept.append((a, b, weight, shared))
        return kept
