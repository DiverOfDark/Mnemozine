"""Structural tests for the docker-compose deployment (PRD Deliverable #1).

These are pure-YAML / static-text assertions plus an optional `docker compose
config` validation gated behind binary availability — so the suite passes fully
offline with no Docker, FalkorDB, Ollama or Qwen running.

What we assert:
  * every PRD-mandated service is present (FalkorDB, Ollama, a Qwen extraction
    LLM, the LiteLLM gateway, the MCP server, the ingest daemon, the maintenance
    scheduler);
  * FalkorDB has a named persistence volume (FR-STO-2 durability);
  * services carry healthchecks;
  * the mnemozine workloads point config.py at the in-network service names, not
    the localhost defaults from .env (the YAML-merge-pitfall regression guard);
  * env is wired from the repo-root .env via env_file.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "deploy" / "docker-compose.yml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

# The seven services PRD Deliverable #1 requires the compose stack to wire.
REQUIRED_SERVICES = {
    "falkordb",
    "ollama",
    "qwen",
    "litellm",
    "mnemozine-mcp",
    "mnemozine-ingest",
    "mnemozine-maintenance",
}

# The three long-running mnemozine workloads built from the shared image.
MNEMOZINE_SERVICES = {
    "mnemozine-mcp": "mnemozine-mcp",
    "mnemozine-ingest": "mnemozine-ingest",
    "mnemozine-maintenance": "mnemozine-maintenance",
}


def _load_compose() -> dict:
    """Parse docker-compose.yml. PyYAML resolves the `<<` merge keys for us."""
    with COMPOSE_FILE.open() as fh:
        return yaml.safe_load(fh)


def test_compose_file_exists() -> None:
    assert COMPOSE_FILE.is_file(), f"missing {COMPOSE_FILE}"


def test_all_required_services_present() -> None:
    compose = _load_compose()
    services = set(compose["services"])
    missing = REQUIRED_SERVICES - services
    assert not missing, f"compose is missing required services: {sorted(missing)}"


def test_falkordb_has_named_persistence_volume() -> None:
    """FR-STO-2: FalkorDB is the single durable store; it MUST persist."""
    compose = _load_compose()
    top_level_volumes = set(compose.get("volumes") or {})
    assert "falkordb-data" in top_level_volumes, "no named FalkorDB volume declared"

    falkordb = compose["services"]["falkordb"]
    mounts = falkordb.get("volumes", [])
    # mounts are "source:target[:opts]" short-form strings here.
    assert any(
        isinstance(m, str) and m.split(":")[0] == "falkordb-data" and ":/data" in m
        for m in mounts
    ), "falkordb does not mount the falkordb-data volume at /data"


def test_falkordb_image_is_pinned_to_verified_tag() -> None:
    """FalkorDB is pinned to an explicit, non-floating tag verified to support the
    index-backed KNN seam FR-RET-2 depends on (``CALL db.idx.vector.queryNodes``).

    The Live phase verified ``v4.18.10`` (graph module ver 41810, Redis 8.6.3 —
    identical digest to ``falkordb/falkordb:latest`` on 2026-06-10) runs the
    queryNodes probe. The tag must NOT float to ``latest``/``edge`` so a deploy
    can't silently land on an image without the vector index. Bump deliberately
    and re-run the probe.
    """
    compose = _load_compose()
    image = compose["services"]["falkordb"]["image"]
    # short-form is `${FALKORDB_IMAGE:-falkordb/falkordb:<tag>}`; take the default.
    default = image.split(":-", 1)[1].rstrip("}") if ":-" in image else image
    assert default == "falkordb/falkordb:v4.18.10", (
        f"FalkorDB image default must be the verified pin, got {default!r}"
    )
    tag = default.rsplit(":", 1)[1]
    assert tag not in {"latest", "edge"}, f"FalkorDB tag must not float: {tag!r}"


def test_every_long_running_service_has_a_healthcheck() -> None:
    """Healthchecks on every long-running service.

    One-shot / on-demand jobs are excluded: ``ollama-init`` (model pull) and
    ``mnemozine-eval`` (runs the §9 scaling assertion once and exits).
    """
    compose = _load_compose()
    one_shot = {"ollama-init", "mnemozine-eval"}
    for name, svc in compose["services"].items():
        if name in one_shot:
            continue
        assert "healthcheck" in svc, f"service {name} has no healthcheck"
        assert "test" in svc["healthcheck"], f"service {name} healthcheck has no test"


def test_eval_service_present_and_runs_eval_cli() -> None:
    """The §9 eval harness is wired as a one-shot service (Goal-5 scaling proof)."""
    compose = _load_compose()
    assert "mnemozine-eval" in compose["services"], "compose missing mnemozine-eval"
    svc = compose["services"]["mnemozine-eval"]
    cmd = svc["command"]
    assert "mnemozine-eval" in cmd, f"mnemozine-eval should run its console script, got {cmd}"
    # Built from the shared image and not auto-started by a plain `up`.
    assert "build" in svc or "image" in svc, "mnemozine-eval has no build/image"
    assert "eval" in svc.get("profiles", []), "mnemozine-eval should be in the 'eval' profile"


def test_mnemozine_services_use_shared_image_build() -> None:
    compose = _load_compose()
    for name in MNEMOZINE_SERVICES:
        svc = compose["services"][name]
        # Built from the repo Dockerfile (multi-stage) OR a prebuilt image ref.
        assert "build" in svc or "image" in svc, f"{name} has no build/image"
        if "build" in svc:
            assert svc["build"]["dockerfile"] == "deploy/Dockerfile"
            assert svc["build"]["context"] == ".."


def test_mnemozine_services_run_correct_console_script() -> None:
    """Each mnemozine workload runs its pyproject console_script."""
    compose = _load_compose()
    for name, script in MNEMOZINE_SERVICES.items():
        svc = compose["services"][name]
        cmd = svc["command"]
        assert script in cmd, f"{name} should run {script}, got {cmd}"


def test_mnemozine_services_wire_env_from_dotenv() -> None:
    compose = _load_compose()
    for name in MNEMOZINE_SERVICES:
        svc = compose["services"][name]
        env_files = svc.get("env_file", [])
        # env_file may be a string or list after merge.
        if isinstance(env_files, str):
            env_files = [env_files]
        assert any("../.env" in str(e) for e in env_files), (
            f"{name} does not load env from the repo-root .env"
        )


def test_mnemozine_services_target_in_network_service_names() -> None:
    """Regression guard for the YAML `<<` merge pitfall.

    `.env.example` ships localhost connection URLs (for bare-metal dev). Inside
    the compose network the mnemozine services must instead reach FalkorDB /
    Ollama / LiteLLM by their compose service names, set under `environment:`
    (which overrides env_file). PyYAML resolves the merge anchors so we read the
    effective per-service map here.
    """
    compose = _load_compose()
    for name in MNEMOZINE_SERVICES:
        env = compose["services"][name].get("environment", {})
        # environment may be a dict (mapping form) — that's what we author.
        assert isinstance(env, dict), f"{name} environment is not a mapping"
        falkor = env.get("MNEMOZINE_FALKORDB__URL", "")
        ollama = env.get("MNEMOZINE_EMBEDDING__BASE_URL", "")
        assert "falkordb" in falkor, f"{name} FalkorDB URL not service-name: {falkor!r}"
        assert "localhost" not in falkor, f"{name} still points FalkorDB at localhost"
        assert "ollama" in ollama, f"{name} Ollama URL not service-name: {ollama!r}"
        assert "localhost" not in ollama, f"{name} still points Ollama at localhost"


def test_mcp_binds_all_interfaces_and_publishes_port() -> None:
    compose = _load_compose()
    mcp = compose["services"]["mnemozine-mcp"]
    env = mcp["environment"]
    assert env.get("MNEMOZINE_MCP_HOST") == "0.0.0.0", "MCP must bind 0.0.0.0 in-container"
    assert mcp.get("ports"), "MCP server publishes no port"


def test_ingest_mounts_claude_transcripts_readonly() -> None:
    """FR-ING-2: the ingest daemon tails Claude Code JSONL transcripts."""
    compose = _load_compose()
    ingest = compose["services"]["mnemozine-ingest"]
    env = ingest["environment"]
    assert env.get("MNEMOZINE_INGEST__CLAUDE_CONFIG_DIR") == "/claude"
    mounts = ingest.get("volumes", [])
    assert any(":/claude:ro" in str(m) for m in mounts), "claude transcripts not mounted ro"


def test_extraction_routed_through_litellm_gateway() -> None:
    """FR-ING-3: extraction goes through the LiteLLM gateway so turns are captured."""
    compose = _load_compose()
    env = compose["services"]["mnemozine-mcp"]["environment"]
    base = env.get("MNEMOZINE_EXTRACTION__BASE_URL", "")
    assert "litellm" in base, f"extraction not routed via litellm gateway: {base!r}"


def test_falkordb_and_ollama_are_dependencies_of_mnemozine_services() -> None:
    compose = _load_compose()
    for name in MNEMOZINE_SERVICES:
        deps = compose["services"][name].get("depends_on", {})
        assert "falkordb" in deps, f"{name} does not depend on falkordb"
        assert "ollama" in deps, f"{name} does not depend on ollama"


def test_persistence_volumes_declared() -> None:
    compose = _load_compose()
    volumes = set(compose.get("volumes") or {})
    # FalkorDB graph+vectors, Ollama models, Qwen model weights all persist.
    for v in ("falkordb-data", "ollama-data", "qwen-models"):
        assert v in volumes, f"missing named volume {v}"


@pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="docker binary not available; skipping live `docker compose config`",
)
def test_docker_compose_config_validates(tmp_path: Path) -> None:
    """`docker compose config` must parse and resolve the stack.

    `docker compose config` reads the repo-root .env (env_file: ../.env); create
    one from .env.example if absent, and clean it up afterwards.
    """
    if shutil.which("docker") is None:  # pragma: no cover - guarded by skipif
        pytest.skip("docker missing")

    dotenv = REPO_ROOT / ".env"
    created = False
    if not dotenv.exists():
        dotenv.write_text(ENV_EXAMPLE.read_text())
        created = True
    try:
        proc = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "config"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:  # pragma: no cover
        pytest.skip(f"docker compose config could not run: {exc}")
    finally:
        if created:
            dotenv.unlink(missing_ok=True)

    assert proc.returncode == 0, f"docker compose config failed:\n{proc.stderr}"
    # Sanity: the resolved output keeps the service-name wiring.
    assert "redis://falkordb:6379" in proc.stdout
