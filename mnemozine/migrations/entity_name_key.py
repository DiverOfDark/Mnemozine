"""The v2 migration — backfill entity ``name_key`` + the normalized-name index.

This is the second concrete :class:`~mnemozine.migrations.Migration`, appended to
the :data:`~mnemozine.migrations.MIGRATIONS` registry at import time. It ships the
storage side of the identity-by-normalized-name fix: every entity node gains a
storage-only ``name_key = toLower(canonical_name)`` property, backed by the
:data:`~mnemozine.storage.graphiti_client.ENTITY_NAME_KEY_INDEX` range index, so
the resolve-or-create-by-name seam
(:meth:`~mnemozine.interfaces.StorageBackend.resolve_or_create_entity`) reuses the
existing node for a normalized name instead of minting a duplicate.

WHAT IT FIXES (the duplicate-entity leak, in place)
---------------------------------------------------
``services._persist`` used to call ``upsert_entity`` (an id-keyed MERGE) for every
extracted entity, minting a fresh node per extraction — so the live store
accumulated thousands of normalized-name-exact duplicate entity nodes. The ingest
path is now unified on ``resolve_or_create_entity`` (identity by
``toLower(canonical_name)``), and this migration backfills the ``name_key`` it
matches on so the seam works against already-stored nodes. (Collapsing the
*existing* duplicates is the separate ``dedup-entities`` catch-up, which must run
AFTER this migration so survivors carry ``name_key``.)

WHY IT IS CHEAP (auto-on-startup safe)
--------------------------------------
:attr:`~mnemozine.migrations.Migration.requires_reextract` is ``False``: it needs
no extractor / GPU / raw transcript — just an idempotent Cypher SET pass over the
entity nodes plus an index create. So the startup hook may auto-apply it.

HOW IT STAMPS (both tiers, FR-MNT-5 idempotency)
------------------------------------------------
Entity nodes carry NO ``data_version`` — the version stamp lives on the MEMORY and
RAW-CHUNK tiers, and
:meth:`~mnemozine.interfaces.StorageBackend.min_data_version` mins over BOTH. So
after the structural ``name_key`` backfill the migration MUST also advance both
tiers to v2 (via
:meth:`~mnemozine.interfaces.StorageBackend.set_data_version` /
:meth:`~mnemozine.interfaces.StorageBackend.set_chunk_data_version`) or
``min_data_version`` never reaches 2 and the step re-runs on every boot. Re-running
over an already-migrated store finds nothing unset and nothing below v2 in either
tier and is a no-op.
"""

from __future__ import annotations

import logging

from mnemozine.interfaces import Extractor, MaintenanceReport, StorageBackend
from mnemozine.migrations.report import MigrationReport

logger = logging.getLogger(__name__)

#: The data-model version this migration produces.
_ENTITY_NAME_KEY_VERSION = 2


class EntityNameKeyMigration:
    """Cheap migration to data_version 2 (entity ``name_key`` backfill + index).

    Satisfies the :class:`~mnemozine.migrations.Migration` Protocol structurally
    (``version`` / ``description`` / ``requires_reextract`` properties + an async
    :meth:`run`). CHEAP (:attr:`requires_reextract` is ``False``) so the startup
    hook may auto-apply it; it ignores any ``extractor`` passed to :meth:`run`.
    """

    @property
    def version(self) -> int:
        return _ENTITY_NAME_KEY_VERSION

    @property
    def description(self) -> str:
        return (
            "v2: backfill name_key=lower(canonical_name) + index on entities "
            "(identity-by-name; stops minting duplicate entity nodes on ingest)"
        )

    @property
    def requires_reextract(self) -> bool:
        # Cheap structural backfill: idempotent Cypher SET + index create, no
        # extractor / GPU. Safe to auto-run at startup.
        return False

    async def run(
        self, backend: StorageBackend, *, extractor: Extractor | None = None
    ) -> MaintenanceReport:
        """Backfill entity ``name_key`` + ensure the index, then stamp both tiers.

        Idempotent (FR-MNT-5): the STRUCTURAL pass
        (:meth:`~mnemozine.interfaces.StorageBackend.backfill_entity_name_keys`)
        ensures the index and only touches entity nodes whose ``name_key`` is unset;
        the TIER-STAMP pass advances every memory + raw chunk it selected below v2
        so :meth:`~mnemozine.interfaces.StorageBackend.min_data_version` actually
        reaches 2. A re-run finds nothing unset and nothing below v2 and is a no-op.
        ``extractor`` is ignored — this is the cheap, no-LLM path.
        """

        del extractor  # cheap path: no re-extraction.
        report = MigrationReport(
            migration="migrate_entity_name_key_v2",
            from_version=self.version - 1,
            to_version=self.version,
        )

        # (1) STRUCTURAL: ensure the index + backfill name_key on unset entities.
        entities_stamped = await backend.backfill_entity_name_keys()

        # (2) TIER-STAMP: advance BOTH tiers to v2 (entity nodes carry no
        # data_version, so the version floor is on memories + chunks). Without this
        # min_data_version never reaches 2 and the migration re-runs every boot.
        memory_ids = [
            memory.id
            async for memory in backend.iter_memories_below_version(self.version)
        ]
        mem_stamped = 0
        if memory_ids:
            mem_stamped = await backend.set_data_version(memory_ids, self.version)

        chunk_hashes = [
            chunk.content_hash
            async for chunk in backend.iter_chunks_below_version(self.version)
        ]
        chunks_stamped = 0
        if chunk_hashes:
            chunks_stamped = await backend.set_chunk_data_version(
                chunk_hashes, self.version
            )

        report.migrated = entities_stamped + mem_stamped + chunks_stamped
        report.notes.append(
            f"backfilled name_key on {entities_stamped} entit(ies); "
            f"stamped {mem_stamped} memor(ies) + {chunks_stamped} raw chunk(s) to "
            f"v{self.version}"
        )
        return report.to_maintenance()


#: The singleton v2 migration instance registered in MIGRATIONS.
ENTITY_NAME_KEY_MIGRATION = EntityNameKeyMigration()


def register() -> None:
    """Append :data:`ENTITY_NAME_KEY_MIGRATION` to the registry (idempotent).

    Mirrors :func:`mnemozine.migrations.baseline.register`: the import-light
    :mod:`mnemozine.migrations.__init__` keeps :data:`~mnemozine.migrations.MIGRATIONS`
    empty (so :mod:`mnemozine.schema.models` can read
    :data:`~mnemozine.migrations.CURRENT_DATA_VERSION` without an import cycle);
    this concrete migration registers itself here, away from that import path.
    Matched by version so a re-import / repeated call never duplicates the step.
    Re-validates the registry invariant after appending so a mis-version fails loud.
    """

    from mnemozine.migrations import MIGRATIONS, validate_migrations

    if any(m.version == ENTITY_NAME_KEY_MIGRATION.version for m in MIGRATIONS):
        return
    MIGRATIONS.append(ENTITY_NAME_KEY_MIGRATION)
    validate_migrations()


# Register on import: importing the runner (or this module) seeds the registry.
register()


__all__ = [
    "ENTITY_NAME_KEY_MIGRATION",
    "EntityNameKeyMigration",
    "register",
]
