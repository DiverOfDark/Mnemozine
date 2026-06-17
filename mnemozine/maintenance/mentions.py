"""Persist (memory)-[:MNEMOZINE_MENTIONS]->(entity) edges from ``m.entities``.

Today a memory's mentioned entities live only as a string-list PROPERTY
(``m.entities``) on the memory node — there are no graph edges between memories
and the entities they name. This maintenance pass turns those names into real,
traversable ``MNEMOZINE_MENTIONS`` edges so the graph becomes navigable
memory<->entity<->memory, which is the substrate the co-mention layer derives
from (two memories that mention the same entity are linkable).

The pass is **set-based and idempotent** (FR-MNT-5): the whole job is a single
``MERGE`` over :meth:`StorageBackend.persist_mentions` (never blind CREATE), so a
re-run asserts exactly the same edges and adds nothing new. The name->entity
resolution is case-folded on both sides (mirroring :meth:`get_entity`) to absorb
the linkage drift between mention names and canonical names.

Implements :class:`~mnemozine.interfaces.MaintenanceJob`. Depends only on the
:class:`~mnemozine.interfaces.StorageBackend` Protocol, so it is unit-testable
offline against the conftest fake.
"""

from __future__ import annotations

import logging

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import MaintenanceReport, StorageBackend

logger = logging.getLogger(__name__)


class MentionsJob:
    """Persist memory->entity mention edges from ``m.entities`` (graph connectivity).

    A thin driver over :meth:`StorageBackend.persist_mentions`: all the work (the
    name resolution + the idempotent MERGE) lives in the backend so the same
    single set-based statement runs against FalkorDB and the in-memory fakes.
    Safe to re-run (FR-MNT-5): the MERGE semantics mean a second pass asserts the
    same edges and creates nothing new.
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
        return "mentions"

    async def run(self) -> MaintenanceReport:
        """Assert the memory->entity mention edges once (idempotent, FR-MNT-5).

        Calls :meth:`StorageBackend.persist_mentions` and records the number of
        edges asserted in :attr:`~mnemozine.interfaces.MaintenanceReport.edges_added`
        plus a one-line note.
        """

        report = MaintenanceReport(job_name=self.name)
        asserted = await self._storage.persist_mentions()
        report.edges_added = asserted
        report.notes.append(
            f"asserted {asserted} memory->entity mention edge(s) "
            f"(MNEMOZINE_MENTIONS, idempotent MERGE)"
        )
        return report
