"""Entity dedup — merge true-duplicate entities, repointing ALL edge types.

The graph fragments when the same real-world entity is stored as several nodes:
``GitHub`` / ``github`` (case drift), ``Rust`` with ``rust-lang`` already in its
aliases, or two near-identical name embeddings. The diagnosis found ~45 groups /
~81 redundant nodes that are *true* duplicates. This job consolidates them in
place over the existing graph by driving the EXISTING
:meth:`~mnemozine.interfaces.StorageBackend.merge_entities` path — the same merge
that already repoints the source's edges onto the survivor (extended for this
feature to repoint ``MNEMOZINE_RELATES`` **and** ``MNEMOZINE_MENTIONS`` **and**
``MNEMOZINE_CO_MENTIONS``, so no edge type is orphaned). No memory is ever
deleted; only true-duplicate ENTITY nodes are folded away.

It is the entity analogue of :class:`~mnemozine.maintenance.category_merge.CategoryMergeJob`,
built on the :class:`~mnemozine.maintenance.entity_resolution.EntityResolutionJob`
template (enumerate -> group -> deterministic survivor -> ``merge_entities``).
Three duplicate-detection ``mode``s, selected by ``graph.entity_dedup_mode`` (the
CLI ``--mode`` overrides it):

* ``exact`` (default) — group only by ``lower(canonical_name)`` collisions. The
  conservative, embedding-free default that catches case/spacing drift and
  nothing else (only true duplicates merge).
* ``alias`` — additionally group when one entity's canonical name (case-folded)
  is already an *alias* of another (the explicit "these are the same thing"
  signal the alias list encodes).
* ``embedding`` — additionally cluster names whose embeddings are within
  ``graph.entity_dedup_similarity_threshold`` cosine (the fuzzier near-dup mode,
  behind the flag; needs an :class:`~mnemozine.interfaces.EmbeddingProvider`,
  mirroring :class:`CategoryMergeJob`).

The survivor of each group is chosen deterministically (reusing
:func:`~mnemozine.maintenance.entity_resolution._pick_survivor`) so the pass is
**idempotent** (FR-MNT-5): once a group is folded into its survivor, a re-run
finds no collision and merges 0.
"""

from __future__ import annotations

import logging

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import EmbeddingProvider, MaintenanceReport, StorageBackend
from mnemozine.maintenance.decision import cosine_similarity
from mnemozine.maintenance.entity_resolution import _pick_survivor
from mnemozine.schema.models import Entity

logger = logging.getLogger(__name__)

#: The duplicate-detection modes EntityDedupJob understands (CLI ``--mode``).
DEDUP_MODES = ("exact", "alias", "embedding")


class EntityDedupJob:
    """Merge true-duplicate entities via the existing ``merge_entities`` path.

    Depends only on :class:`~mnemozine.interfaces.StorageBackend` (enumeration +
    the existing merge) and, for ``mode='embedding'`` only, an optional
    :class:`~mnemozine.interfaces.EmbeddingProvider`. The merge repoints ALL three
    edge types onto the survivor, so the consolidation never orphans a
    relates/mentions/co-mention edge. Idempotent (FR-MNT-5).
    """

    def __init__(
        self,
        storage: StorageBackend,
        *,
        embeddings: EmbeddingProvider | None = None,
        settings: Settings | None = None,
        mode: str | None = None,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._settings = settings or get_settings()
        # CLI --mode overrides graph.entity_dedup_mode for this run.
        self._mode = (mode or self._settings.graph.entity_dedup_mode).strip().lower()

    @property
    def name(self) -> str:
        return "entity_dedup"

    async def run(self) -> MaintenanceReport:
        """Group true duplicates and fold each non-survivor into its survivor.

        Records the number of entities merged in
        :attr:`~mnemozine.interfaces.MaintenanceReport.entities_merged` and a
        per-group note. Idempotent: a second run finds no duplicate group and
        merges 0.
        """

        report = MaintenanceReport(job_name=self.name)
        if self._mode not in DEDUP_MODES:
            report.notes.append(
                f"unknown entity_dedup mode {self._mode!r}; expected one of "
                f"{', '.join(DEDUP_MODES)} — no entities merged"
            )
            return report

        entities = [e async for e in self._storage.iter_entities()]
        groups = await self._group_duplicates(entities)

        merged = 0
        for survivor, dups in groups:
            for dup in dups:
                await self._storage.merge_entities(dup.id, survivor.id)
                merged += 1
            report.notes.append(
                f"merged {len(dups)} duplicate entit(ies) into "
                f"'{survivor.canonical_name}' (mode={self._mode})"
            )
        report.entities_merged = merged
        if merged:
            report.notes.append(
                f"entity dedup ({self._mode}): merged {merged} duplicate entit(ies) "
                f"into {len(groups)} survivor(s)"
            )
        return report

    # --- grouping (per-mode; deterministic) --------------------------------

    async def _group_duplicates(
        self, entities: list[Entity]
    ) -> list[tuple[Entity, list[Entity]]]:
        """Return ``(survivor, [non-survivors])`` for each duplicate group.

        ``exact`` groups by ``lower(canonical_name)``; ``alias`` additionally
        unions groups linked by an alias membership; ``embedding`` additionally
        unions groups whose names are within the cosine threshold. The survivor of
        each group is the deterministic :func:`_pick_survivor`, so the orientation
        (and thus the whole pass) is stable across runs (idempotency, FR-MNT-5).
        """

        if not entities:
            return []

        # Stable, deterministic enumeration order regardless of store iteration.
        ordered = sorted(entities, key=lambda e: (e.canonical_name.lower(), e.id))

        # Start from exact lower(canonical_name) groups (a union-find keyed by the
        # representative entity id), then widen per mode.
        clusters: dict[str, list[Entity]] = {}
        for e in ordered:
            clusters.setdefault(e.canonical_name.strip().lower(), []).append(e)

        if self._mode in ("alias", "embedding"):
            clusters = self._merge_by_alias(ordered, clusters)
        if self._mode == "embedding":
            clusters = await self._merge_by_embedding(ordered, clusters)

        groups: list[tuple[Entity, list[Entity]]] = []
        for members in clusters.values():
            if len(members) < 2:
                continue
            survivor = _pick_survivor(members)
            dups = sorted(
                (m for m in members if m.id != survivor.id),
                key=lambda e: (e.canonical_name.lower(), e.id),
            )
            if dups:
                groups.append((survivor, dups))
        # Deterministic group order (by survivor) so the report/merges are stable.
        groups.sort(key=lambda g: (g[0].canonical_name.lower(), g[0].id))
        return groups

    def _merge_by_alias(
        self, ordered: list[Entity], clusters: dict[str, list[Entity]]
    ) -> dict[str, list[Entity]]:
        """Union clusters when one entity's canonical name is another's alias.

        The alias list is the explicit "these are the same thing" signal, so an
        entity whose canonical name (case-folded) appears in another entity's
        aliases is folded into that entity's cluster.
        """

        # key (lower canonical) -> the cluster key it currently belongs to.
        key_of: dict[str, str] = {}
        for ckey, members in clusters.items():
            for m in members:
                key_of[m.canonical_name.strip().lower()] = ckey

        # alias (lower) -> set of cluster keys that declare it.
        for e in ordered:
            ekey = key_of.get(e.canonical_name.strip().lower())
            if ekey is None:
                continue
            for alias in e.aliases:
                other_key = key_of.get(alias.strip().lower())
                if other_key is not None and other_key != ekey:
                    self._union(clusters, key_of, other_key, ekey)
        return {k: v for k, v in clusters.items() if v}

    async def _merge_by_embedding(
        self, ordered: list[Entity], clusters: dict[str, list[Entity]]
    ) -> dict[str, list[Entity]]:
        """Union clusters whose representative names embed within the threshold.

        Mirrors :class:`CategoryMergeJob`: embeds each entity's canonical name once
        (best-effort) and unions two clusters when any cross-cluster name pair is
        at or above ``graph.entity_dedup_similarity_threshold`` cosine. A missing
        provider or a failed embed simply skips this widening (the exact/alias
        groups still stand).
        """

        if self._embeddings is None:
            return clusters
        threshold = self._settings.graph.entity_dedup_similarity_threshold

        key_of: dict[str, str] = {}
        for ckey, members in clusters.items():
            for m in members:
                key_of[m.canonical_name.strip().lower()] = ckey

        vectors = await self._embed_names(ordered)
        names = [e for e in ordered if e.canonical_name in vectors]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                ka = key_of.get(a.canonical_name.strip().lower())
                kb = key_of.get(b.canonical_name.strip().lower())
                if ka is None or kb is None or ka == kb:
                    continue
                sim = cosine_similarity(
                    vectors[a.canonical_name], vectors[b.canonical_name]
                )
                if sim >= threshold:
                    self._union(clusters, key_of, kb, ka)
        return {k: v for k, v in clusters.items() if v}

    @staticmethod
    def _union(
        clusters: dict[str, list[Entity]],
        key_of: dict[str, str],
        src_key: str,
        dst_key: str,
    ) -> None:
        """Fold cluster ``src_key`` into ``dst_key`` (in place), updating ``key_of``."""

        if src_key == dst_key:
            return
        moved = clusters.pop(src_key, [])
        clusters.setdefault(dst_key, []).extend(moved)
        for m in moved:
            key_of[m.canonical_name.strip().lower()] = dst_key

    # --- embedding helper (best-effort) ------------------------------------

    async def _embed_names(self, entities: list[Entity]) -> dict[str, list[float]]:
        if self._embeddings is None:
            return {}
        vectors: dict[str, list[float]] = {}
        for e in entities:
            try:
                vectors[e.canonical_name] = await self._embeddings.embed(
                    e.canonical_name
                )
            except Exception:  # noqa: BLE001 - degrade to no embedding widening
                logger.warning(
                    "entity dedup embed failed for %r; skipping embedding widening",
                    e.canonical_name,
                    exc_info=True,
                )
        return vectors
