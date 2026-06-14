"""Structural tests for the docker-compose deployment (PRD Deliverable #1).

These are pure-YAML / static-text assertions plus an optional `docker compose
config` validation gated behind binary availability — so the suite passes fully
offline with no Docker, FalkorDB or Ollama running.

The stack collapses to THREE long-running services on a plain `up`:
  * ``falkordb``  — the single graph + vector store,
  * ``ollama``    — bge-m3 embeddings AND a qwen extraction model (its OpenAI
    ``/v1`` endpoint is the default extraction backend),
  * ``mnemozine`` — the all-in-one app (runs the ``mnemozine`` console script =
    ``mnemozine.app:run_all``: MCP + ingest + maintenance + web under one loop).

Plus one one-shot helper (``ollama-init``) and OPTIONAL profile-gated services
(``litellm`` under ``gateway``, ``qwen`` under ``qwen-llamacpp``,
``mnemozine-eval`` under ``eval``) that a plain ``up`` does NOT start.

A second standalone compose file (``docker-compose.ingest.yml``) runs ONLY a
``mnemozine-ingest`` service for splitting ingest onto a different machine
against a remote store.

What we assert:
  * the three default long-running services are present and nothing extra is
    auto-started by a plain ``up`` (optional services are profile-gated);
  * FalkorDB has a named persistence volume (FR-STO-2 durability);
  * long-running services carry healthchecks;
  * the all-in-one app points config.py at the in-network service names, not the
    localhost defaults from .env (the YAML-merge-pitfall regression guard);
  * extraction targets the Ollama OpenAI-compatible endpoint by default;
  * the all-in-one publishes 8765 (WebUI + mounted /mcp) and mounts ~/.claude ro;
  * the split-ingest override exists, runs the standalone ingest script, and
    does NOT override the remote endpoints (they come from .env);
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
INGEST_COMPOSE_FILE = REPO_ROOT / "deploy" / "docker-compose.ingest.yml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

# The three long-running services a plain `up` starts.
DEFAULT_LONG_RUNNING_SERVICES = {"falkordb", "ollama", "mnemozine"}

# One-shot helper started by a plain `up` but which exits when done.
ONE_SHOT_SERVICES = {"ollama-init"}

# Optional, profile-gated services NOT started by a plain `up`.
PROFILE_GATED_SERVICES = {
    "litellm": "gateway",
    "qwen": "qwen-llamacpp",
    "mnemozine-eval": "eval",
}


def _load_compose(path: Path = COMPOSE_FILE) -> dict:
    """Parse a compose file. PyYAML resolves the `<<` merge keys for us."""
    with path.open() as fh:
        return yaml.safe_load(fh)


def test_compose_file_exists() -> None:
    assert COMPOSE_FILE.is_file(), f"missing {COMPOSE_FILE}"


def test_ingest_override_file_exists() -> None:
    assert INGEST_COMPOSE_FILE.is_file(), f"missing {INGEST_COMPOSE_FILE}"


def test_default_up_starts_only_three_long_running_services() -> None:
    """A plain `up` (no --profile) starts exactly falkordb + ollama + mnemozine.

    Every other service must be one-shot (`restart: "no"`) or hidden behind a
    profile, so the stack collapses to ~3 containers (the App-phase contract).
    """
    compose = _load_compose()
    services = compose["services"]

    # The three long-running services are present and have no profile gate.
    for name in DEFAULT_LONG_RUNNING_SERVICES:
        assert name in services, f"compose missing default service {name!r}"
        assert not services[name].get("profiles"), (
            f"{name} must NOT be profile-gated; it starts on a plain `up`"
        )

    # Anything started by a plain `up` (no profile) that is long-running must be
    # one of those three. Profile-gated services are excluded by Compose; the
    # only other no-profile service allowed is the one-shot ollama-init.
    auto_started = {
        name
        for name, svc in services.items()
        if not svc.get("profiles")
    }
    allowed = DEFAULT_LONG_RUNNING_SERVICES | ONE_SHOT_SERVICES
    extra = auto_started - allowed
    assert not extra, f"plain `up` would also start unexpected services: {sorted(extra)}"

    # The one-shot helper does not stay up.
    assert services["ollama-init"].get("restart") == "no"


def test_optional_services_are_profile_gated() -> None:
    """litellm (gateway), qwen (qwen-llamacpp) and eval are opt-in via profiles."""
    compose = _load_compose()
    services = compose["services"]
    for name, profile in PROFILE_GATED_SERVICES.items():
        assert name in services, f"compose missing optional service {name!r}"
        profiles = services[name].get("profiles", [])
        assert profile in profiles, (
            f"{name} must be in the {profile!r} profile, got {profiles}"
        )


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


def test_all_in_one_uses_shared_image_build() -> None:
    """The all-in-one app is built from the repo Dockerfile (multi-stage)."""
    compose = _load_compose()
    svc = compose["services"]["mnemozine"]
    assert "build" in svc or "image" in svc, "mnemozine has no build/image"
    if "build" in svc:
        assert svc["build"]["dockerfile"] == "deploy/Dockerfile"
        assert svc["build"]["context"] == ".."


def test_all_in_one_runs_the_mnemozine_console_script() -> None:
    """The all-in-one app runs the new `mnemozine` console script (run_all)."""
    compose = _load_compose()
    cmd = compose["services"]["mnemozine"]["command"]
    assert "mnemozine" in cmd, f"all-in-one should run `mnemozine`, got {cmd}"
    # Must run the ALL-IN-ONE script, not a single-component one.
    for single in ("mnemozine-mcp", "mnemozine-ingest", "mnemozine-maintenance", "mnemozine-web"):
        assert single not in cmd, f"all-in-one must run `mnemozine`, not {single}: {cmd}"


def test_all_in_one_has_graceful_shutdown() -> None:
    """run_all traps SIGTERM; the all-in-one uses compose's default stop signal."""
    compose = _load_compose()
    svc = compose["services"]["mnemozine"]
    # stop_signal is optional (SIGTERM is the compose default), but if set it must
    # be SIGTERM/SIGINT which run_all handles for a graceful drain.
    stop_signal = svc.get("stop_signal", "SIGTERM")
    assert stop_signal in {"SIGTERM", "SIGINT"}, f"unexpected stop_signal {stop_signal!r}"


def test_all_in_one_wires_env_from_dotenv() -> None:
    compose = _load_compose()
    svc = compose["services"]["mnemozine"]
    env_files = svc.get("env_file", [])
    if isinstance(env_files, str):
        env_files = [env_files]
    assert any("../.env" in str(e) for e in env_files), (
        "the all-in-one does not load env from the repo-root .env"
    )


def test_all_in_one_targets_in_network_service_names() -> None:
    """Regression guard for the YAML `<<` merge pitfall.

    `.env.example` ships localhost connection URLs (for bare-metal dev). Inside
    the compose network the all-in-one app must instead reach FalkorDB / Ollama
    by their compose service names, set under `environment:` (which overrides
    env_file). PyYAML resolves the merge anchors so we read the effective
    per-service map here.
    """
    compose = _load_compose()
    env = compose["services"]["mnemozine"].get("environment", {})
    assert isinstance(env, dict), "mnemozine environment is not a mapping"
    falkor = env.get("MNEMOZINE_FALKORDB__URL", "")
    ollama = env.get("MNEMOZINE_EMBEDDING__BASE_URL", "")
    assert "falkordb" in falkor, f"FalkorDB URL not service-name: {falkor!r}"
    assert "localhost" not in falkor, "still points FalkorDB at localhost"
    assert "ollama" in ollama, f"Ollama URL not service-name: {ollama!r}"
    assert "localhost" not in ollama, "still points Ollama at localhost"


def test_extraction_targets_ollama_openai_endpoint() -> None:
    """Default extraction is served by Ollama's OpenAI-compatible /v1 endpoint.

    The separate qwen + litellm services are no longer required by default. Because
    we hit Ollama's OpenAI-compatible ``/v1`` path (not Ollama's native API), the
    LiteLLM model id MUST be prefixed ``openai/`` and the base_url MUST carry the
    ``/v1`` suffix. (LiteLLM's ``ollama/`` provider speaks Ollama's native /api/*
    surface and would 404 against /v1 — verified live.)
    """
    compose = _load_compose()
    env = compose["services"]["mnemozine"]["environment"]
    base = env.get("MNEMOZINE_EXTRACTION__BASE_URL", "")
    model = env.get("MNEMOZINE_EXTRACTION__MODEL", "")
    assert "ollama" in base, f"extraction not pointed at ollama: {base!r}"
    assert "litellm" not in base, f"extraction must NOT default to litellm: {base!r}"
    assert base.rstrip("}").endswith("/v1"), (
        f"Ollama OpenAI endpoint needs the /v1 suffix: {base!r}"
    )
    assert "openai/" in model, (
        f"extraction model must be a LiteLLM openai/ id for Ollama's /v1 path "
        f"(ollama/ 404s against /v1): {model!r}"
    )


def test_all_in_one_publishes_8765_and_binds_all_interfaces() -> None:
    """Single port 8765 serves the WebUI + mounted /mcp; bind 0.0.0.0 in-container."""
    compose = _load_compose()
    svc = compose["services"]["mnemozine"]
    env = svc["environment"]
    assert env.get("MNEMOZINE_WEB__HOST") == "0.0.0.0", "web must bind 0.0.0.0 in-container"
    ports = svc.get("ports", [])
    assert any("8765" in str(p) for p in ports), f"all-in-one must publish 8765: {ports}"


def test_all_in_one_mounts_claude_transcripts_readonly() -> None:
    """FR-ING-2: the ingest component tails Claude Code JSONL transcripts."""
    compose = _load_compose()
    svc = compose["services"]["mnemozine"]
    env = svc["environment"]
    assert env.get("MNEMOZINE_INGEST__CLAUDE_CONFIG_DIR") == "/claude"
    mounts = svc.get("volumes", [])
    assert any(":/claude:ro" in str(m) for m in mounts), "claude transcripts not mounted ro"


def test_all_in_one_depends_on_falkordb_and_ollama_healthy() -> None:
    compose = _load_compose()
    deps = compose["services"]["mnemozine"].get("depends_on", {})
    assert "falkordb" in deps, "all-in-one does not depend on falkordb"
    assert "ollama" in deps, "all-in-one does not depend on ollama"
    # depends_on uses the long form with a health condition.
    for dep in ("falkordb", "ollama"):
        cond = deps[dep]
        if isinstance(cond, dict):
            assert cond.get("condition") == "service_healthy", (
                f"all-in-one should wait for {dep} to be healthy"
            )


def test_ollama_init_pulls_both_embedding_and_qwen_models() -> None:
    """ollama-init seeds BOTH bge-m3 (embeddings) AND a qwen extraction model."""
    compose = _load_compose()
    init = compose["services"]["ollama-init"]
    cmd = init["command"]
    text = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
    assert "bge-m3" in text, f"ollama-init does not pull bge-m3: {text!r}"
    assert "qwen" in text.lower(), f"ollama-init does not pull a qwen model: {text!r}"
    # Two pulls -> two `ollama pull` invocations.
    assert text.count("ollama pull") >= 2, (
        f"ollama-init should pull BOTH models: {text!r}"
    )


def test_persistence_volumes_declared() -> None:
    compose = _load_compose()
    volumes = set(compose.get("volumes") or {})
    # FalkorDB graph+vectors, Ollama models (bge-m3 + qwen), optional qwen GGUF.
    for v in ("falkordb-data", "ollama-data", "qwen-models"):
        assert v in volumes, f"missing named volume {v}"


# --------------------------------------------------------------------------- #
# Split-ingest override: deploy/docker-compose.ingest.yml                      #
# --------------------------------------------------------------------------- #


def test_ingest_override_has_only_the_ingest_service() -> None:
    """The split-ingest compose is standalone: ONLY a mnemozine-ingest service."""
    compose = _load_compose(INGEST_COMPOSE_FILE)
    services = set(compose["services"])
    assert services == {"mnemozine-ingest"}, (
        f"ingest override should define only mnemozine-ingest, got {sorted(services)}"
    )


def test_ingest_override_runs_standalone_ingest_script() -> None:
    """It runs the standalone `mnemozine-ingest` console script (one component)."""
    compose = _load_compose(INGEST_COMPOSE_FILE)
    svc = compose["services"]["mnemozine-ingest"]
    cmd = svc["command"]
    assert "mnemozine-ingest" in cmd, f"should run mnemozine-ingest, got {cmd}"


def test_ingest_override_reads_remote_endpoints_from_dotenv() -> None:
    """The remote endpoints come straight from .env — NOT overridden here.

    The whole point of split-ingest is to point at REMOTE homelab endpoints, so
    the override must load ../.env and must NOT pin FalkorDB / Ollama / extraction
    URLs to in-network compose service names under `environment:`.
    """
    compose = _load_compose(INGEST_COMPOSE_FILE)
    svc = compose["services"]["mnemozine-ingest"]

    env_files = svc.get("env_file", [])
    if isinstance(env_files, str):
        env_files = [env_files]
    assert any("../.env" in str(e) for e in env_files), (
        "ingest override does not load env from the repo-root .env"
    )

    env = svc.get("environment", {}) or {}
    # The remote endpoints MUST NOT be hardcoded to compose service names; they
    # are supplied by .env so the operator points them at the homelab.
    for key in (
        "MNEMOZINE_FALKORDB__URL",
        "MNEMOZINE_EMBEDDING__BASE_URL",
        "MNEMOZINE_EXTRACTION__BASE_URL",
    ):
        assert key not in env, (
            f"{key} must come from .env (remote), not be overridden in the "
            f"split-ingest override"
        )


def test_ingest_override_mounts_claude_transcripts_readonly() -> None:
    """FR-ING-2: the split-ingest daemon tails Claude Code JSONL transcripts ro."""
    compose = _load_compose(INGEST_COMPOSE_FILE)
    svc = compose["services"]["mnemozine-ingest"]
    env = svc.get("environment", {})
    assert env.get("MNEMOZINE_INGEST__CLAUDE_CONFIG_DIR") == "/claude"
    mounts = svc.get("volumes", [])
    assert any(":/claude:ro" in str(m) for m in mounts), "claude transcripts not mounted ro"


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


@pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="docker binary not available; skipping live `docker compose config`",
)
def test_ingest_override_config_validates() -> None:
    """`docker compose config` must parse the split-ingest override too."""
    if shutil.which("docker") is None:  # pragma: no cover - guarded by skipif
        pytest.skip("docker missing")

    dotenv = REPO_ROOT / ".env"
    created = False
    if not dotenv.exists():
        dotenv.write_text(ENV_EXAMPLE.read_text())
        created = True
    try:
        proc = subprocess.run(
            ["docker", "compose", "-f", str(INGEST_COMPOSE_FILE), "config"],
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
