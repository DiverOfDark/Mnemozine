"""OQ3 — the embedding index / re-embed migration (``mnemozine-maintenance migrate-index``).

When the embedding model (and therefore the vector **dimension**) changes, the
fixed-dimension FalkorDB vector index over ``MnemozineMemory.embedding`` can no
longer index the existing vectors: a FalkorDB vector index is created at a single
``OPTIONS {dimension: N}`` width and cannot be resized in place. Resolving OQ3
(PRD §10 / FR-MNT-3 re-embed) therefore requires a maintenance pass that:

1. **detects** a configured-vs-actual dimension mismatch — compares
   ``embedding.dimensions`` (config) against the live index width read back from
   FalkorDB (``GraphitiClient.current_vector_index_dimension``);
2. **drops + recreates** the vector index at the new width
   (``GraphitiClient.recreate_vector_index``, which the Config phase confirmed is
   the existing ``ensure_vector_index`` seam plus a matching drop); and
3. **re-embeds every hot memory** through the :class:`EmbeddingProvider` by
   iterating :meth:`StorageBackend.iter_memories` (hot tier) and calling
   :meth:`StorageBackend.reembed` per unit (FR-MNT-3: full background re-embed of
   the hot tier on a model change; the archive tier is re-embedded lazily on
   promotion, already handled by ``StorageBackend.promote``).

No new ``StorageBackend`` Protocol method is needed (per the Config phase's
interface note): the migration is covered by ``embedding.dimensions`` +
``GraphitiClient.ensure_vector_index`` + ``StorageBackend.iter_memories`` +
``StorageBackend.reembed``, all already in the contract.

Idempotency / re-run safety (FR-MNT-5): when the live width already equals the
configured width the pass is a **no-op** by default (no drop, no re-embed) so it
is cheap and safe to run on every maintenance cycle. ``force=True`` re-embeds the
hot tier even without a dimension change (useful when only the *model* changed but
kept the same width, so the index is fine but the vectors are stale). Re-embedding
is itself idempotent (``reembed`` recomputes deterministically), so a re-run after
a crash mid-pass simply finishes the job.
"""

from __future__ import annotations

import logging
from typing import Protocol

from mnemozine.config import Settings, get_settings
from mnemozine.interfaces import (
    EmbeddingProvider,
    MaintenanceReport,
    StorageBackend,
)
from mnemozine.schema.models import Tier

logger = logging.getLogger(__name__)


class VectorIndexAdmin(Protocol):
    """The FalkorDB vector-index admin seam the migration needs (OQ3).

    Implemented by :class:`mnemozine.storage.graphiti_client.GraphitiClient`. Kept
    as a narrow Protocol so :class:`MigrateIndexJob` depends only on these three
    operations (and can be unit-tested against a fake admin), not on the concrete
    client / a live FalkorDB.
    """

    async def current_vector_index_dimension(self) -> int | None:
        """Live width of the memory vector index, or ``None`` if absent."""
        ...

    async def recreate_vector_index(self) -> None:
        """Drop + recreate the memory vector index at the configured width."""
        ...

    @property
    def embedding_dimensions(self) -> int:
        """The width the client (re)creates the index at — the config target."""
        ...


def needs_migration(
    *, configured_dim: int, actual_dim: int | None, force: bool = False
) -> bool:
    """Decide whether the index/re-embed migration must run (OQ3 decision logic).

    Pure + side-effect free so it is trivially unit-testable. Rules:

    * ``force`` -> always migrate (re-embed even if the width is unchanged).
    * ``actual_dim is None`` -> the index does not exist yet (or its width could
      not be read): **no migration**. A fresh store builds the index at the right
      width on ``connect()``; there are no stale vectors to fix, and an
      undeterminable width is treated conservatively as "leave it alone" rather
      than dropping a possibly-correct index.
    * ``actual_dim != configured_dim`` -> the live index is the wrong width:
      **migrate** (drop+recreate+re-embed).
    * otherwise (widths match) -> **no migration** (idempotent no-op).
    """

    if force:
        return True
    if actual_dim is None:
        return False
    return actual_dim != configured_dim


class MigrateIndexJob:
    """OQ3 index/re-embed migration as a re-runnable maintenance pass (FR-MNT-5).

    Structurally a :class:`mnemozine.interfaces.MaintenanceJob` (``name`` +
    ``run``) so it can be reported on like the other passes, but it is *not* part
    of the default scheduled set — re-embedding the whole hot tier is an explicit,
    operator-triggered migration (run after an ``embedding.model`` change), wired
    as the ``migrate-index`` subcommand of ``mnemozine-maintenance``.
    """

    def __init__(
        self,
        storage: StorageBackend,
        index_admin: VectorIndexAdmin,
        embeddings: EmbeddingProvider,
        *,
        settings: Settings | None = None,
        force: bool = False,
    ) -> None:
        self._storage = storage
        self._admin = index_admin
        self._embeddings = embeddings
        self._settings = settings or get_settings()
        self._force = force

    @property
    def name(self) -> str:
        return "migrate-index"

    async def run(self) -> MaintenanceReport:
        """Detect a dimension change, recreate the index, and re-embed the hot tier.

        Returns a :class:`MaintenanceReport` whose ``consolidated`` count is reused
        to report the number of memories re-embedded (the report has no dedicated
        field and adding one would be a cross-stream interface change), with the
        decision + dimensions recorded in ``notes`` for the audit log.
        """

        configured_dim = self._settings.embedding.dimensions
        # Cross-check the provider's own dimensionality against config so a
        # mis-set ``embedding.dimensions`` is visible in the audit log rather than
        # silently producing an index the provider's vectors won't fit.
        provider_dim = self._embeddings.dimensions

        actual_dim = await self._admin.current_vector_index_dimension()
        migrate = needs_migration(
            configured_dim=configured_dim, actual_dim=actual_dim, force=self._force
        )

        notes = [
            f"configured_dim={configured_dim}",
            f"actual_index_dim={actual_dim}",
            f"provider_dim={provider_dim}",
            f"force={self._force}",
            f"migrate={migrate}",
        ]
        if provider_dim != configured_dim:
            notes.append(
                "WARNING: embedding provider dimensions "
                f"({provider_dim}) != configured embedding.dimensions "
                f"({configured_dim}); the recreated index uses the configured width."
            )

        if not migrate:
            logger.info("migrate-index: no dimension change; nothing to do (%s)", notes)
            return MaintenanceReport(job_name=self.name, notes=notes)

        # 1) Drop + recreate the fixed-dimension vector index at the new width.
        await self._admin.recreate_vector_index()
        notes.append("recreated vector index")

        # 2) Re-embed every hot memory so the new (correct-width) vectors land in
        #    the freshly-created index. Archive tier is re-embedded lazily on
        #    promotion (StorageBackend.promote), so it is intentionally skipped.
        reembedded = 0
        async for mem in self._storage.iter_memories(tier=Tier.HOT):
            await self._storage.reembed(mem.id)
            reembedded += 1
        notes.append(f"reembedded_hot={reembedded}")
        logger.info("migrate-index: recreated index and re-embedded %d hot memories", reembedded)

        return MaintenanceReport(
            job_name=self.name, consolidated=reembedded, notes=notes
        )
