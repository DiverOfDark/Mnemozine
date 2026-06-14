"""The Mnemozine operator-console WebUI (FastAPI) — WEBUI PRD.

A **local, single-operator** JSON API + static-SPA host over the existing
composition root (:class:`mnemozine.app.Container` -> ``StorageBackend``,
retriever, maintenance jobs, evals, activity log). The UI is read-first with a
small set of HITL mutations; it is **never a new source of truth** — every route
goes through the existing layers.

Public surface (the backend contract Phase-2 and the frontend depend on):

* :func:`create_app` — the app factory: builds the FastAPI app from a
  ``Container``, installs auth (optional static bearer token), locked CORS, the
  full router set, and the static-SPA mount. Importing this package does not
  require a live FalkorDB; ``create_app`` is pure construction (lazy backend).
* :mod:`mnemozine.web.schemas` — the pydantic wire models (the contract).
* :mod:`mnemozine.web.deps`    — request-scoped dependency providers over the
  Container.
"""

from __future__ import annotations

from mnemozine.web.app import create_app

__all__ = ["create_app"]
