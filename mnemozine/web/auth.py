"""Optional static-bearer-token auth for the WebUI API (WEBUI PRD Q5).

The console is local-only and binds to localhost by default, so auth is
**optional**: when ``web.token`` is unset the API is open on the bound interface
(fine for a localhost bind). When it is set, every ``/api`` request must carry a
matching ``Authorization: Bearer <token>`` header (or ``?token=`` query param as
a convenience for the SPA's static fetches), else 401.

This is intentionally a single static token — there is no multi-user/RBAC (PRD
§7 out of scope). The check is constant-time to avoid a token-comparison timing
oracle.
"""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status

from mnemozine.web.deps import ContainerDep


def _extract_token(request: Request) -> str | None:
    """Pull a bearer token from the Authorization header or ``?token=`` query."""

    auth = request.headers.get("authorization")
    if auth:
        scheme, _, value = auth.partition(" ")
        if scheme.lower() == "bearer" and value:
            return value.strip()
    qp = request.query_params.get("token")
    return qp.strip() if qp else None


async def require_auth(request: Request, container: ContainerDep) -> None:
    """FastAPI dependency: enforce the static bearer token when configured (Q5).

    No-op when ``web.token`` is unset (open API on a localhost bind). When set,
    rejects any request whose presented token does not match, with a 401.
    """

    configured = container.settings.web.token
    if not configured:
        return
    presented = _extract_token(request)
    if presented is None or not hmac.compare_digest(presented, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


__all__ = ["require_auth"]
