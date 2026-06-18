"""Data-versioning + in-place migration framework — the shared CONTRACT.

When the data model / extraction / scope-derivation changes, **existing data is
migrated IN PLACE** — never wiped and re-ingested (re-ingest is hours of GPU).
There are two migration strategies, both already provided as StorageBackend
seams, that a concrete :class:`Migration` composes:

* **cheap reclassify** — re-derive scope/category/cross-ref from the
  already-stored content + provenance (no raw transcript needed), via
  :meth:`~mnemozine.interfaces.StorageBackend.reclassify_memory`. Survives
  Claude's 30-day local cleanup (R4).
* **re-extract from retained raw chunks** — re-run a newer extractor/classifier
  over the durable :class:`~mnemozine.schema.models.RawChunk` tier, via
  :meth:`~mnemozine.interfaces.StorageBackend.re_extract_from_raw_chunks`.

VERSION STAMP
-------------
Every :class:`~mnemozine.schema.models.MemoryUnit` and
:class:`~mnemozine.schema.models.RawChunk` carries an integer ``data_version``,
defaulted to :data:`CURRENT_DATA_VERSION` and stamped at write time. Records
written before this feature (or with no field) are treated as version **0** (see
:func:`record_data_version`). Bumping :data:`CURRENT_DATA_VERSION` and appending
a :class:`Migration` to :data:`MIGRATIONS` is the entire surface for shipping a
data-model change.

IDEMPOTENCY (FR-MNT-5)
----------------------
A :class:`Migration` only touches records with ``data_version < migration.version``
and stamps each touched record up to ``migration.version``.

Because :meth:`~mnemozine.interfaces.StorageBackend.min_data_version` takes the min
over BOTH the memory tier and the raw-chunk tier, a migration must advance EVERY
tier it selected, not just the memory tier — otherwise a stale chunk keeps
``min_data_version`` below the target, ``pending_migrations`` keeps reporting the
step, and with ``migrate.auto_on_startup`` it re-runs on every boot. So a migration
must:

* stamp the MEMORY records it selected (via
  :meth:`~mnemozine.interfaces.StorageBackend.iter_memories_below_version`) up to
  its version — explicitly via
  :meth:`~mnemozine.interfaces.StorageBackend.set_data_version`, or implicitly
  because ``reclassify_memory`` / ``re_extract_from_raw_chunks`` re-stamp the
  touched record to :data:`CURRENT_DATA_VERSION`; AND
* stamp the RAW-CHUNK records it selected (via
  :meth:`~mnemozine.interfaces.StorageBackend.iter_chunks_below_version`) up to its
  version — explicitly via
  :meth:`~mnemozine.interfaces.StorageBackend.set_chunk_data_version` even on the
  *cheap* reclassify path, or implicitly because
  ``re_extract_from_raw_chunks`` re-stamps each re-processed chunk on the *heavy*
  path.

A cheap reclassify migration therefore still calls ``set_chunk_data_version`` on the
chunks it selected so the chunk tier reaches the new version without the GPU
re-extract. Re-running a migration whose version is already reached is then a true
no-op (both tiers are already at or above it).

This module is the single home of :data:`CURRENT_DATA_VERSION` and the
:data:`MIGRATIONS` registry. It is import-light on purpose (no top-level import of
``schema``/``interfaces``) so that :mod:`mnemozine.schema.models` can read the
constant for its field default without an import cycle; the heavier types used by
the :class:`Migration` Protocol are imported lazily / under ``TYPE_CHECKING``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # import-cycle-safe: these are only needed for type hints.
    from mnemozine.interfaces import Extractor, MaintenanceReport, StorageBackend

# ---------------------------------------------------------------------------
# The single source of truth for the current data-model version.
# ---------------------------------------------------------------------------

#: The data-model version the current code writes. Stamped onto every
#: :class:`~mnemozine.schema.models.MemoryUnit` / :class:`RawChunk` at write time
#: (their ``data_version`` field defaults to this). Bump this by 1 whenever a
#: data-model / extraction / scope-derivation change needs an in-place migration,
#: and append a :class:`Migration` producing the new version to :data:`MIGRATIONS`.
CURRENT_DATA_VERSION: int = 2

#: The version assigned to records written before this feature (no ``data_version``
#: field) or that were explicitly never stamped. Migrations select on
#: ``data_version < target``, so unstamped records (version 0) are always in scope
#: for the first migration. See :func:`record_data_version`.
UNSTAMPED_DATA_VERSION: int = 0


def record_data_version(value: Any) -> int:
    """Normalize a possibly-missing/None ``data_version`` to a concrete int.

    Records written before this feature have no ``data_version`` property; reading
    one back yields ``None`` (or a missing key). Treat any such record as
    :data:`UNSTAMPED_DATA_VERSION` (0) so it is always in scope for the first
    migration. A real integer is returned unchanged. Storage backends should route
    the value they read off a node through this helper when computing
    :meth:`~mnemozine.interfaces.StorageBackend.min_data_version`.
    """

    if value is None:
        return UNSTAMPED_DATA_VERSION
    try:
        return int(value)
    except (TypeError, ValueError):
        return UNSTAMPED_DATA_VERSION


# ---------------------------------------------------------------------------
# The Migration contract.
# ---------------------------------------------------------------------------


@runtime_checkable
class Migration(Protocol):
    """One ordered, idempotent in-place data migration step (the CONTRACT).

    A migration takes the store from ``version - 1`` to its own :attr:`version`.
    Concrete migrations live downstream (they fill :meth:`run`'s behavior); this
    package only defines the shape and the :data:`MIGRATIONS` registry so the
    runner, the WebUI, and the startup hook can all code against it.

    Idempotency rule (FR-MNT-5): :meth:`run` MUST only touch records whose
    ``data_version`` is strictly less than :attr:`version`, and MUST stamp every
    record it selected — in BOTH tiers it touched — up to :attr:`version`:

    * MEMORY tier: select via
      :meth:`~mnemozine.interfaces.StorageBackend.iter_memories_below_version`;
      stamp implicitly (``reclassify_memory`` / ``re_extract_from_raw_chunks``
      re-stamp to :data:`CURRENT_DATA_VERSION`) or explicitly via
      :meth:`~mnemozine.interfaces.StorageBackend.set_data_version`.
    * RAW-CHUNK tier: select via
      :meth:`~mnemozine.interfaces.StorageBackend.iter_chunks_below_version`; stamp
      implicitly on the heavy path (``re_extract_from_raw_chunks`` re-stamps each
      re-processed chunk) or explicitly via
      :meth:`~mnemozine.interfaces.StorageBackend.set_chunk_data_version` on the
      cheap path.

    Because :meth:`~mnemozine.interfaces.StorageBackend.min_data_version` mins over
    both tiers, a cheap reclassify migration that leaves a stale chunk behind would
    never reach its target version and would re-run on every boot. It MUST therefore
    advance its selected chunks via ``set_chunk_data_version`` even though it does
    not re-extract them. Re-running a migration whose :attr:`version` is already
    reached must be a no-op (both tiers already at or above it).
    """

    @property
    def version(self) -> int:
        """The data-model version this migration PRODUCES (the target).

        After a successful :meth:`run`, every previously-eligible record is at
        ``data_version >= version``. Must be unique and monotonically increasing
        across :data:`MIGRATIONS`, and ``<= CURRENT_DATA_VERSION``. This invariant
        is enforced at import time by :func:`validate_migrations` (the registry must
        be the contiguous sequence ``1, 2, 3, ..., CURRENT_DATA_VERSION``), so a
        mis-registered migration fails loudly instead of mis-ordering the apply set.
        """
        ...

    @property
    def description(self) -> str:
        """Human-readable one-line summary (shown in logs / the WebUI)."""
        ...

    @property
    def requires_reextract(self) -> bool:
        """Cost/kind marker: True for a HEAVY re-extract-from-raw-chunks migration.

        Distinguishes the two migration strategies so the startup hook can honor the
        config contract (``migrate.auto_on_startup`` runs CHEAP migrations only;
        heavy re-extract migrations are *never* auto-run regardless of that flag —
        see :class:`~mnemozine.config.MigrateSettings`):

        * ``False`` — a CHEAP *reclassify* migration: re-derives
          scope/category/cross-ref from already-stored content and stamps both tiers
          in place (``reclassify_memory`` / ``set_data_version`` /
          ``set_chunk_data_version``). No extractor / GPU needed; safe to auto-run.
        * ``True`` — a HEAVY migration that re-runs an extractor over the raw tier
          (``re_extract_from_raw_chunks``); requires the ``extractor`` arg and GPU
          time. The startup hook MUST skip these when auto-applying and leave them
          to the operator-triggered ``re-extract`` path.
        """
        ...

    async def run(
        self, backend: StorageBackend, *, extractor: Extractor | None = None
    ) -> MaintenanceReport:
        """Apply this migration in place over ``backend`` (idempotent).

        Selects records below :attr:`version` in BOTH tiers — memories via
        :meth:`~mnemozine.interfaces.StorageBackend.iter_memories_below_version` and
        raw chunks via
        :meth:`~mnemozine.interfaces.StorageBackend.iter_chunks_below_version` — and
        migrates them via the cheap-reclassify seam
        (:meth:`~mnemozine.interfaces.StorageBackend.reclassify_memory`, then
        :meth:`~mnemozine.interfaces.StorageBackend.set_chunk_data_version` to
        advance the stale chunks it selected) and/or the re-extract-from-raw-chunks
        seam (:meth:`~mnemozine.interfaces.StorageBackend.re_extract_from_raw_chunks`,
        which re-stamps chunks implicitly). It then stamps each touched record in
        both tiers up to :attr:`version`, so
        :meth:`~mnemozine.interfaces.StorageBackend.min_data_version` actually
        reaches the target. ``extractor`` is required only when
        :attr:`requires_reextract` is True (migrations that re-extract from the raw
        tier); reclassify-only migrations ignore it. Returns a
        :class:`~mnemozine.interfaces.MaintenanceReport` summarizing the pass
        (reuse its ``notes`` for per-step detail; see :class:`MigrationReport` for
        the from/to-version convenience shape).
        """
        ...


# ---------------------------------------------------------------------------
# The ordered registry.
# ---------------------------------------------------------------------------

#: The ordered list of migrations, ascending by :attr:`Migration.version`. The
#: runner applies, in order, every migration whose ``version`` is greater than the
#: store's current :meth:`~mnemozine.interfaces.StorageBackend.min_data_version`
#: (the "pending" set). Empty until a data-model change ships its first migration;
#: downstream appends concrete migrations here. Keep it sorted and gap-free
#: (1, 2, 3, ...) up to :data:`CURRENT_DATA_VERSION`. Enforced by
#: :func:`validate_migrations` (called at import time and from the startup hook).
MIGRATIONS: list[Migration] = []


class MigrationRegistryError(ValueError):
    """Raised when :data:`MIGRATIONS` violates the registry invariant.

    The registry must be the contiguous, gap-free, duplicate-free sequence of
    versions ``1, 2, 3, ..., CURRENT_DATA_VERSION`` (see
    :func:`validate_migrations`). A duplicate, a gap, a non-positive version, or a
    version above :data:`CURRENT_DATA_VERSION` would mis-order or skip a migration
    step, so it is a hard, fail-loud configuration error rather than a silent
    mis-apply.
    """


def validate_migrations(
    migrations: list[Migration] | None = None,
    *,
    current_version: int | None = None,
) -> list[Migration]:
    """Validate the migration registry invariant, returning it sorted by version.

    The invariant (previously prose-only): the registered migration versions must
    be **unique, each positive and ``<= CURRENT_DATA_VERSION``, and contiguous /
    gap-free** — i.e. exactly ``1, 2, 3, ..., CURRENT_DATA_VERSION`` (or a prefix
    of it). Any violation raises :class:`MigrationRegistryError` so a
    mis-registered migration fails loudly at import/startup instead of silently
    corrupting the apply order or skipping a step. Returns the migrations sorted
    ascending by :attr:`Migration.version` (the canonical apply order) so callers
    can use the validated, ordered list directly.

    ``migrations`` / ``current_version`` default to the module-level
    :data:`MIGRATIONS` / :data:`CURRENT_DATA_VERSION` (so the import-time guard and
    the startup hook both call it with no args); the parameters exist so tests can
    validate a candidate registry in isolation.
    """

    regs = MIGRATIONS if migrations is None else migrations
    current = CURRENT_DATA_VERSION if current_version is None else current_version

    versions = [m.version for m in regs]
    # Each version must be a positive int and within the current data version.
    for v in versions:
        if not isinstance(v, int) or v <= 0:
            raise MigrationRegistryError(
                f"migration version must be a positive int, got {v!r}"
            )
        if v > current:
            raise MigrationRegistryError(
                f"migration version {v} exceeds CURRENT_DATA_VERSION {current}"
            )
    # Unique.
    if len(set(versions)) != len(versions):
        dupes = sorted({v for v in versions if versions.count(v) > 1})
        raise MigrationRegistryError(
            f"duplicate migration version(s) in registry: {dupes}"
        )
    # Contiguous / gap-free: sorted versions must be exactly 1..len(regs).
    expected = list(range(1, len(versions) + 1))
    if sorted(versions) != expected:
        raise MigrationRegistryError(
            f"migration registry must be contiguous {expected!r} (1..N), "
            f"got {sorted(versions)!r}"
        )
    return sorted(regs, key=lambda m: m.version)


def pending_migrations(current_version: int) -> list[Migration]:
    """Return the migrations that still need to run, in apply order.

    Given the store's lowest ``data_version`` (its
    :meth:`~mnemozine.interfaces.StorageBackend.min_data_version`), return every
    registered migration whose :attr:`Migration.version` is strictly greater,
    ascending. An empty list means the store is fully migrated (nothing pending).
    Pure / side-effect-free so the startup hook and the WebUI can show "N pending"
    without touching the store twice. Validates the registry invariant first (via
    :func:`validate_migrations`) so a mis-registered migration is caught before its
    versions are used to order/skip steps.
    """

    ordered = validate_migrations()
    return [m for m in ordered if m.version > current_version]


# Import-time guard: a mis-registered migration (duplicate / gap / over-current
# version) must fail loudly the moment this module is imported, before any
# apply-order decision is made. ``MIGRATIONS`` is empty by default, which is valid
# (no-op), so this is a no-op until a concrete migration is appended.
validate_migrations()


__all__ = [
    "CURRENT_DATA_VERSION",
    "MIGRATIONS",
    "Migration",
    "MigrationRegistryError",
    "UNSTAMPED_DATA_VERSION",
    "pending_migrations",
    "record_data_version",
    "validate_migrations",
]
