"""All WebUI API routers (WEBUI PRD §6), registered on the app by ``create_app``.

Each router covers one screen's data and is mounted under ``/api`` with the
optional bearer-token dependency. In this Phase-1 foundation the route bodies are
typed STUBS that return schema-valid sample/empty data: the goal is a bootable
app with a **complete OpenAPI** so the frontend foundation and Phase-2 backend
fillers code against a frozen contract. Phase-2 fills the bodies against the real
``StorageBackend`` / retriever / maintenance / evals — the signatures and
response schemas here are the contract and must not drift.

Routers (PRD §6): :mod:`memories`, :mod:`graph`, :mod:`recall`, :mod:`crossrefs`,
:mod:`activity`, :mod:`maintenance`, :mod:`eval`, :mod:`mutations`, :mod:`health`.
"""

from __future__ import annotations

from mnemozine.web.routes import (
    activity,
    crossrefs,
    eval,
    graph,
    health,
    maintenance,
    memories,
    mutations,
    recall,
)

# The ordered router set create_app mounts. Each module exposes ``router``.
ALL_ROUTERS = [
    health.router,
    memories.router,
    mutations.router,
    graph.router,
    recall.router,
    crossrefs.router,
    activity.router,
    maintenance.router,
    eval.router,
]

__all__ = ["ALL_ROUTERS"]
