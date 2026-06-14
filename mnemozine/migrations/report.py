"""The :class:`MigrationReport` value object for a migration pass.

A migration ultimately reports through the existing
:class:`~mnemozine.interfaces.MaintenanceReport` (so the maintenance runner,
activity feed, and WebUI render it uniformly with every other pass). This module
adds a thin, migration-flavored convenience report that also records the
from/to ``data_version`` of the pass; :meth:`MigrationReport.to_maintenance`
collapses it back to a plain ``MaintenanceReport`` for those shared sinks.

Kept in its own module (not ``migrations/__init__``) so importing
:data:`~mnemozine.migrations.CURRENT_DATA_VERSION` stays import-light and free of
the ``interfaces`` import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mnemozine.interfaces import MaintenanceReport


@dataclass(slots=True)
class MigrationReport:
    """Summary of one in-place migration pass (FR-MNT-5, R5 audit).

    ``migrated`` is the number of records (memories + chunks) actually touched and
    re-stamped. ``from_version`` / ``to_version`` record the data-model versions
    the pass moved the in-scope records between (``to_version`` equals the
    migration's :attr:`~mnemozine.migrations.Migration.version`). ``notes`` carries
    per-step detail for the audit log, exactly like
    :class:`~mnemozine.interfaces.MaintenanceReport.notes`.
    """

    migration: str
    from_version: int
    to_version: int
    migrated: int = 0
    notes: list[str] = field(default_factory=list)

    def to_maintenance(self) -> MaintenanceReport:
        """Collapse to a :class:`~mnemozine.interfaces.MaintenanceReport`.

        Lets a migration report through the same maintenance sinks (runner log,
        activity feed, WebUI) as every other pass. The migrated count maps onto
        ``re_extracted`` (the closest existing counter — a migration re-derives /
        re-extracts records), and the from/to versions are prepended as a note.
        """

        return MaintenanceReport(
            job_name=self.migration,
            re_extracted=self.migrated,
            notes=[
                f"migrated data_version {self.from_version} -> {self.to_version}",
                *self.notes,
            ],
        )


__all__ = ["MigrationReport"]
