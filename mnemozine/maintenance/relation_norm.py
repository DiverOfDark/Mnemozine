"""Relation normalization — collapse the fragmented RELATES label vocabulary.

The relationship extractor emits a FREE-FORM relation label on every
``MNEMOZINE_RELATES`` edge (``uses`` / ``used-in`` / ``used_in`` /
``depends on`` / ``depends_on`` / ``requires`` …), so the relation registry
fragments over time exactly the way the emergent-category registry and the
entity graph do: the same semantic relation splits across spelling/punctuation
variants, scattering edges that should share one canonical label. This is the
**relation analogue of category merge** (the
:class:`~mnemozine.maintenance.category_merge.CategoryMergeJob`) and of entity
resolution (FR-MNT-4): a periodic, idempotent maintenance pass that

1. enumerates the in-use relation labels with their active-edge counts
   (:meth:`StorageBackend.list_relations` — the relation registry);
2. maps each raw label to its canonical form through a **controlled vocabulary**
   (:func:`normalize_relation` + the module-level :data:`RELATION_SYNONYMS` map),
   not a similarity threshold — the vocabulary is a code-level constant like the
   entity-suffix list, so the collapse is deterministic rather than
   embedding-gated;
3. folds every non-canonical label into its canonical one via
   :meth:`StorageBackend.merge_relations`, combining the parallel edges' weights
   (``max``, matching :meth:`StorageBackend.upsert_edge`'s re-assert semantics)
   and deleting the now-redundant parallel source edge so no duplicate parallel
   edges remain between the same entity pair + canonical relation.

Because the mapping is a fixed vocabulary, the pass is **idempotent** (FR-MNT-5):
once every label is canonical a second run finds nothing to merge (each label
already maps to itself, and ``merge_relations(x, x)`` is a no-op).

:class:`RelationNormJob` implements :class:`~mnemozine.interfaces.MaintenanceJob`
(``name`` / ``run``) and exposes a read-only :meth:`~RelationNormJob.propose_merges`
(the ``--dry-run`` preview, mirroring
:meth:`~mnemozine.maintenance.category_merge.CategoryMergeJob.propose_merges`). It
depends only on the :class:`~mnemozine.interfaces.StorageBackend` Protocol, so it
is unit-testable offline against the conftest fakes.
"""

from __future__ import annotations

import logging
import re

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import MaintenanceReport, StorageBackend

logger = logging.getLogger(__name__)

_NORMALIZE_RE = re.compile(r"[\s\-]+")

# Controlled-vocabulary synonym map: variant spelling/punctuation -> canonical
# relation label. Keys are already in the :func:`normalize_relation` slug form
# (lowercased, hyphen/space collapsed to ``_``, trivial plural/-ing lemmatized),
# so a raw label is looked up by FIRST slugging it and THEN consulting this map.
# This is the relation analogue of the entity-suffix list in
# :func:`~mnemozine.maintenance.entity_resolution.normalize_entity_key`: a
# code-level vocabulary, not a config threshold, so the collapse is deterministic.
RELATION_SYNONYMS: dict[str, str] = {
    # uses / used-in / used_in / using -> uses
    "used_in": "uses",
    "use": "uses",
    "using": "uses",
    "utilizes": "uses",
    "utilizing": "uses",
    "uses_the": "uses",
    # depends-on / depends_on / requires -> depends_on
    "depends": "depends_on",
    "depends_upon": "depends_on",
    "requires": "depends_on",
    "needs": "depends_on",
    "relies_on": "depends_on",
    "relies_upon": "depends_on",
    # composites on / composites-on -> composites_on
    "composites": "composites_on",
    # part-of / part_of / belongs-to -> part_of
    "belongs_to": "part_of",
    "member_of": "part_of",
    "contained_in": "part_of",
    # contains / includes / has -> contains
    "include": "contains",
    "includes": "contains",
    "has": "contains",
    "have": "contains",
    # implements / implemented-by stays implements
    "implemented_by": "implements",
    # related-to / relates-to -> related_to
    "relates_to": "related_to",
    "related": "related_to",
    "associated_with": "related_to",
    # extends / inherits-from -> extends
    "inherits_from": "extends",
    "inherits": "extends",
    "subclass_of": "extends",
    # produces / generates -> produces
    "generates": "produces",
    "creates": "produces",
    # references / refers-to -> references
    "refers_to": "references",
    "refer_to": "references",
}

# The canonical controlled vocabulary: every value the synonym map can collapse
# to, plus the bare canonical labels themselves. A normalized label already in
# this set is left untouched (it is its own canonical form). Used by callers that
# want to know whether a label is canonical without consulting the synonym map.
CONTROLLED_RELATIONS: frozenset[str] = frozenset(RELATION_SYNONYMS.values())


def normalize_relation(label: str) -> str:
    """Normalize a free-form relation label to its canonical controlled form.

    Two stages: (1) slug the label — lowercase, trim, collapse any run of
    whitespace/hyphens to a single ``_``, and lemmatize a trivial trailing
    plural (``-s``) or gerund (``-ing``) so ``Uses`` / ``used`` / ``using`` all
    reach the same slug; (2) map the slug through :data:`RELATION_SYNONYMS` to
    its canonical label (returning the slug unchanged when it has no synonym).

    Deterministic and offline — the relation analogue of
    :func:`~mnemozine.maintenance.category_merge.normalize_category` and
    :func:`~mnemozine.maintenance.entity_resolution.normalize_entity_key`. An
    empty / whitespace-only label falls back to ``"relates"`` (the default edge
    relation), so a degenerate label never produces an empty canonical.
    """

    slug = _NORMALIZE_RE.sub("_", label.strip().lower()).strip("_")
    if not slug:
        return "relates"
    # A label that is ALREADY a canonical relation is a fixed point — short-circuit
    # before the stemmer so ``contains`` does not get plural-stripped to ``contain``
    # and lose idempotency (the canonical set is the authority, FR-MNT-5).
    if slug in CONTROLLED_RELATIONS:
        return slug
    # Map the raw slug through the synonym table FIRST so a known inflected form
    # (``using`` / ``includes``) is collapsed exactly; only when it has no synonym
    # do we lemmatize a trailing plural and retry, so the deterministic vocabulary
    # always wins over the shallow stemmer.
    if slug in RELATION_SYNONYMS:
        return RELATION_SYNONYMS[slug]
    lemma = _lemmatize(slug)
    return RELATION_SYNONYMS.get(lemma, lemma)


def _lemmatize(slug: str) -> str:
    """Fold a trivial trailing plural on the LAST token of a slug.

    Only the final ``_``-segment is touched, and only a trailing ``s`` is
    stripped (``uses`` -> ``use``, ``includes`` -> ``include``) when it leaves a
    non-trivial stem and is not a doubled ``ss``. Gerund (``-ing``) and other
    inflections are handled by the explicit :data:`RELATION_SYNONYMS` map rather
    than a stemmer, since ``-ing`` stripping is error-prone (``using`` -> ``us``).
    Kept deliberately shallow — the heavy lifting is the synonym map.
    """

    head, sep, tail = slug.rpartition("_")
    token = tail
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        token = token[:-1]
    return f"{head}{sep}{token}" if sep else token


class RelationNormJob:
    """Collapse fragmented ``MNEMOZINE_RELATES`` labels into a controlled vocabulary.

    The relation analogue of
    :class:`~mnemozine.maintenance.category_merge.CategoryMergeJob`. Satisfies
    :class:`~mnemozine.interfaces.MaintenanceJob`. Deterministic and
    embedding-free: the canonical mapping is the code-level
    :func:`normalize_relation` / :data:`RELATION_SYNONYMS`, so the same input
    always proposes the same merges and a re-run is idempotent (FR-MNT-5).
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
        return "relation_norm"

    # --- read-only proposal (the --dry-run preview) ------------------------

    async def propose_merges(self) -> list[tuple[str, str]]:
        """Propose ``(source_relation, canonical_relation)`` merges; no writes.

        Pure / read-only (reads only :meth:`StorageBackend.list_relations`): for
        every in-use label whose normalized canonical form differs from the label
        itself, emit a ``source -> canonical`` pair. Deterministic — sorted by
        source label so the preview and the applied order are stable.
        """

        registry = await self._storage.list_relations()
        proposals: list[tuple[str, str]] = []
        for raw, _count in registry:
            canonical = normalize_relation(raw)
            if canonical != raw:
                proposals.append((raw, canonical))
        proposals.sort()
        return proposals

    # --- apply (MaintenanceJob.run) ----------------------------------------

    async def run(self) -> MaintenanceReport:
        """Apply the proposed relation merges once (idempotent, FR-MNT-5).

        Calls :meth:`StorageBackend.merge_relations` for each ``source ->
        canonical`` proposal and records the relabelled-edge count in
        :attr:`~mnemozine.interfaces.MaintenanceReport.relations_merged` plus the
        per-label detail in ``notes``. A re-run over already-canonical labels
        proposes nothing and merges 0.
        """

        report = MaintenanceReport(job_name=self.name)
        proposals = await self.propose_merges()
        relabelled = 0
        for source, target in proposals:
            n = await self._storage.merge_relations(source, target)
            relabelled += n
            report.notes.append(
                f"normalized relation '{source}' -> '{target}' "
                f"({n} edge(s) relabelled)"
            )
        report.relations_merged = relabelled
        if proposals:
            report.notes.append(
                f"normalized {len(proposals)} relation label(s) "
                f"({relabelled} edge(s) relabelled)"
            )
        return report
