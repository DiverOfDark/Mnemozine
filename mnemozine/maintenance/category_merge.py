"""Category merge / consolidation — fold near-duplicate emergent categories.

The classifier emits a FREE-FORM :attr:`~mnemozine.schema.models.MemoryUnit.category`
string (no fixed enum, core data-model redesign), so the category registry
fragments over time the same way the entity graph does:
``gotcha`` / ``gotchas`` / ``pitfall`` all mean the same thing but split the
memories that should cluster together. This is the **category analogue of entity
resolution** (FR-MNT-4): a periodic, idempotent maintenance pass that

1. enumerates the in-use categories with their active-memory counts
   (:meth:`StorageBackend.list_categories` — the category registry);
2. clusters near-duplicates by *similarity of the category names* (embedding
   cosine over the name, plus a cheap string-similarity fallback so a missing /
   degenerate embedding still catches ``gotcha`` vs ``gotchas``) **and** the
   memories filed under them;
3. picks the canonical survivor of each cluster deterministically (highest
   memory count, then shortest, then lexicographically smallest name) and folds
   every other member into it via :meth:`StorageBackend.merge_categories`.

The merge similarity cutoff is ``category.merge_similarity_threshold`` (config,
not a constant — §6.6). The pass is **idempotent** (FR-MNT-5): once a cluster has
been folded into its canonical category a second run finds a single-member
cluster and does nothing.

:class:`CategoryMergeJob` implements both :class:`~mnemozine.interfaces.MaintenanceJob`
(``name`` / ``run``) and the :class:`~mnemozine.interfaces.CategoryMerger`
Protocol (adds :meth:`propose_merges`, a read-only proposal the WebUI can review
before applying). It depends only on the
:class:`~mnemozine.interfaces.StorageBackend` and
:class:`~mnemozine.interfaces.EmbeddingProvider` Protocols, so it is unit-testable
offline against the conftest fakes.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import EmbeddingProvider, MaintenanceReport, StorageBackend
from mnemozine.maintenance.decision import cosine_similarity
from mnemozine.schema.models import DEFAULT_CATEGORY

logger = logging.getLogger(__name__)


def normalize_category(name: str) -> str:
    """Normalize a free-form category to its stable comparison slug.

    Mirrors :class:`~mnemozine.schema.models.MemoryUnit` category normalization
    (lowercased / whitespace-trimmed), with an empty value falling back to
    :data:`~mnemozine.schema.models.DEFAULT_CATEGORY`. Kept here (rather than
    constructing a ``MemoryUnit``) so the merge policy can normalize a bare
    category string with no I/O.
    """

    slug = name.strip().lower()
    return slug or DEFAULT_CATEGORY


def name_similarity(a: str, b: str) -> float:
    """Cheap string similarity of two category *names* in ``[0, 1]``.

    The embedding-free fallback: a ``SequenceMatcher`` ratio over the normalized
    names so plural/spelling variants (``gotcha`` vs ``gotchas``) cluster even
    when no :class:`EmbeddingProvider` is wired or it returns a degenerate
    vector. Deterministic and offline.
    """

    return SequenceMatcher(None, normalize_category(a), normalize_category(b)).ratio()


class CategoryMergeJob:
    """Merge near-duplicate emergent categories into a canonical one (FR-MNT-2/4).

    The category analogue of :class:`~mnemozine.maintenance.entity_resolution.EntityResolutionJob`.
    Satisfies both :class:`~mnemozine.interfaces.MaintenanceJob` and
    :class:`~mnemozine.interfaces.CategoryMerger`.

    Similarity is the **max** of the embedding cosine over the category names and
    a string-similarity ratio, so a degenerate embedding never *lowers* a clear
    textual match. The canonical survivor of a cluster is the highest-count
    category (the one most worth keeping), breaking ties deterministically by
    shortest then lexicographically-smallest name for idempotency (FR-MNT-5).
    """

    def __init__(
        self,
        storage: StorageBackend,
        *,
        embeddings: EmbeddingProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._settings = settings or get_settings()

    @property
    def name(self) -> str:
        return "category_merge"

    # --- read-only proposal (CategoryMerger.propose_merges) ----------------

    async def propose_merges(self) -> list[tuple[str, str]]:
        """Propose ``(source_category, canonical_category)`` merges; no writes.

        Pure / read-only (reads only :meth:`StorageBackend.list_categories` and,
        when available, the embedding provider) so the WebUI can show the
        proposals for review before the operator applies them. Each pair is
        oriented source -> canonical (the higher-count category is the target).
        Deterministic: clusters are formed in descending-count order and the
        survivor chosen by the stable :meth:`_pick_canonical` rule.
        """

        registry = await self._storage.list_categories()
        # Normalize + fold any duplicate raw spellings that share a slug, summing
        # their counts so the registry we cluster over is already canonical-keyed.
        counts: dict[str, int] = {}
        for raw, count in registry:
            counts[normalize_category(raw)] = counts.get(
                normalize_category(raw), 0
            ) + int(count)
        categories = list(counts.items())
        if len(categories) < 2:
            return []

        threshold = self._settings.category.merge_similarity_threshold
        # Precompute name embeddings once per category (best-effort).
        vectors = await self._embed_names([c for c, _ in categories])

        # Process highest-count first so the canonical target of a cluster is the
        # most-populated category and the orientation is stable.
        ordered = sorted(categories, key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
        clusters: list[list[str]] = []
        assigned: set[str] = set()
        for name, _count in ordered:
            if name in assigned:
                continue
            cluster = [name]
            assigned.add(name)
            for other, _other_count in ordered:
                if other in assigned:
                    continue
                if self._similar(name, other, vectors, threshold):
                    cluster.append(other)
                    assigned.add(other)
            clusters.append(cluster)

        proposals: list[tuple[str, str]] = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            canonical = self._pick_canonical(cluster, counts)
            for member in cluster:
                if member != canonical:
                    proposals.append((member, canonical))
        return proposals

    # --- apply (MaintenanceJob.run) ----------------------------------------

    async def run(self) -> MaintenanceReport:
        """Apply the proposed category merges once (idempotent, FR-MNT-5).

        Calls :meth:`StorageBackend.merge_categories` for each proposed pair and
        records the re-labelled count in
        :attr:`~mnemozine.interfaces.MaintenanceReport.categories_merged` and the
        per-pair detail in ``notes``.
        """

        report = MaintenanceReport(job_name=self.name)
        proposals = await self.propose_merges()
        merged_categories = 0
        relabelled = 0
        for source, target in proposals:
            n = await self._storage.merge_categories(source, target)
            merged_categories += 1
            relabelled += n
            report.notes.append(
                f"merged category '{source}' -> '{target}' ({n} memor(ies) re-labelled)"
            )
        report.categories_merged = merged_categories
        if merged_categories:
            report.notes.append(
                f"merged {merged_categories} categor(ies) "
                f"({relabelled} memor(ies) re-labelled)"
            )
        return report

    # --- helpers -----------------------------------------------------------

    async def _embed_names(self, names: list[str]) -> dict[str, list[float]]:
        """Embed each category *name* once (empty dict if no provider/failure)."""

        if self._embeddings is None:
            return {}
        vectors: dict[str, list[float]] = {}
        for name in names:
            try:
                vectors[name] = await self._embeddings.embed(name)
            except Exception:  # noqa: BLE001 - fall back to string similarity
                logger.warning(
                    "category embed failed for %r; using string similarity", name,
                    exc_info=True,
                )
        return vectors

    def _similar(
        self,
        a: str,
        b: str,
        vectors: dict[str, list[float]],
        threshold: float,
    ) -> bool:
        """True if categories ``a`` and ``b`` are near-duplicates above ``threshold``.

        Similarity is the max of (embedding cosine over the names, if both were
        embedded) and the string-similarity ratio, so the textual fallback can
        only *raise* confidence, never mask a clear lexical match.
        """

        sim = name_similarity(a, b)
        va = vectors.get(a)
        vb = vectors.get(b)
        if va is not None and vb is not None:
            sim = max(sim, cosine_similarity(va, vb))
        return sim >= threshold

    @staticmethod
    def _pick_canonical(cluster: list[str], counts: dict[str, int]) -> str:
        """Deterministically choose a cluster's canonical (survivor) category.

        Prefers the most-populated category (most memories already filed under
        it), breaking ties by shortest then lexicographically-smallest name so
        the choice is stable across runs (idempotency, FR-MNT-5).
        """

        return sorted(
            cluster,
            key=lambda c: (-counts.get(c, 0), len(c), c),
        )[0]
