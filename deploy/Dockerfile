# syntax=docker/dockerfile:1.7
#
# Mnemozine image — multi-stage build using uv for fast, reproducible installs.
#
# Stage 1 (builder): resolve + install the project and its dependencies into a
#   self-contained virtualenv at /opt/venv. We install from the project metadata
#   (pyproject.toml) so the pinned Graphiti / FalkorDB / LiteLLM / MCP stack lands
#   exactly as declared in [project.dependencies].
# Stage 2 (runtime): a slim Python base that only carries the built venv and the
#   package source. No build toolchain in the final image.
#
# The same image backs every Mnemozine service; the docker-compose / Helm layers
# pick the console_script to run via `command:` (mnemozine-mcp / mnemozine-ingest
# / mnemozine-maintenance / mnemozine-eval), all declared in pyproject [project.scripts].

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS builder

# uv: a fast, drop-in pip/venv replacement. Copied from the official image so we
# do not depend on a network curl bootstrap.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build deps some wheels may need to compile (kept out of the runtime image).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create the target virtualenv up front so installs land in a relocatable prefix.
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# 1) Install dependencies first (better layer cache): copy only the metadata.
#    README is referenced by [project].readme so it must be present for the
#    build backend to resolve project metadata.
COPY pyproject.toml README.md ./

# Install just the third-party dependencies (not the project itself yet) so this
# expensive layer is cached across source-only changes. We extract the runtime
# deps from pyproject and install them; the project is installed in the next step.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
      "graphiti-core[falkordb]==0.29.2" \
      "falkordb>=1.6.1" \
      "litellm>=1.88.1" \
      "mcp>=1.27.0" \
      "pydantic>=2.9" \
      "pydantic-settings>=2.14" \
      "ollama>=0.4" \
      "httpx>=0.27" \
      "watchfiles>=0.24" \
      "apscheduler>=3.10,<4" \
      "typer>=0.26"

# 2) Now copy the source and install the project itself (provides the
#    console_scripts: mnemozine-mcp / -ingest / -maintenance / -eval).
COPY mnemozine ./mnemozine
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps .

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

# Minimal runtime: curl backs HTTP HEALTHCHECK probes (MCP); procps provides
# pgrep for the process-liveness probes of the no-HTTP daemons (ingest,
# maintenance); tini is PID 1 for clean signal/zombie handling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl procps tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 mnemozine

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Bring over the fully-built virtualenv and the package source.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/mnemozine /app/mnemozine
COPY pyproject.toml README.md /app/

WORKDIR /app
USER mnemozine

# tini as PID 1 so long-running services (watcher, scheduler, MCP) reap children
# and forward signals cleanly under both docker-compose and Kubernetes.
ENTRYPOINT ["tini", "--"]

# Default command: the MCP server. docker-compose / Helm override `command` per
# service (mnemozine-ingest, mnemozine-maintenance, ...).
CMD ["mnemozine-mcp"]
