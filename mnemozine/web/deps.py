"""Request-scoped dependency providers over the :class:`Container` (WEBUI).

FastAPI routes declare what they need (the container, settings, the storage
backend, the retriever, the activity log) via these providers, so the route
bodies stay thin and the Container remains the single composition root — the UI
is never a new source of truth (WEBUI PRD §2).

The :class:`Container` is stashed on ``app.state`` by :func:`create_app` and read
back here. The storage/retriever/activity providers are **async** because
``Container.build_storage`` opens the FalkorDB connection lazily; routes that
only need config (health stub, schema) depend on the cheap providers and never
force a connection. Phase-1 stubs depend on these but mostly return sample data,
so a route can be exercised without a live backend.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from mnemozine.app import Container
from mnemozine.config import Settings
from mnemozine.interfaces import (
    ActivityLog,
    CrossReferencer,
    Retriever,
    StorageBackend,
)


def get_container(request: Request) -> Container:
    """Return the process-wide :class:`Container` stashed on ``app.state``."""

    container: Container | None = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - create_app always sets it
        raise RuntimeError("Container not configured on app.state (use create_app).")
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def get_settings_dep(container: ContainerDep) -> Settings:
    """Return the live :class:`Settings` from the container."""

    return container.settings


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


async def get_storage(container: ContainerDep) -> StorageBackend:
    """Return the connected storage backend (opens FalkorDB lazily, memoized)."""

    return await container.build_storage()


StorageDep = Annotated[StorageBackend, Depends(get_storage)]


async def get_retriever(container: ContainerDep) -> Retriever:
    """Return the scoped retriever over the connected backend (FR-RET-*)."""

    return await container.build_retriever()


RetrieverDep = Annotated[Retriever, Depends(get_retriever)]


async def get_cross_referencer(container: ContainerDep) -> CrossReferencer:
    """Return the cross-reference engine (FR-RET-6)."""

    return await container.build_cross_referencer()


CrossReferencerDep = Annotated[CrossReferencer, Depends(get_cross_referencer)]


async def get_activity_log(container: ContainerDep) -> ActivityLog:
    """Return the activity log (NullActivityLog unless web.enable_activity_log)."""

    return await container.build_activity_log()


ActivityLogDep = Annotated[ActivityLog, Depends(get_activity_log)]


__all__ = [
    "get_container",
    "ContainerDep",
    "SettingsDep",
    "StorageDep",
    "RetrieverDep",
    "CrossReferencerDep",
    "ActivityLogDep",
]
