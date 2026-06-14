"""Structural + (optional) live tests for the Mnemozine Helm chart.

Static YAML checks run everywhere (offline). When the `helm` binary exists we
additionally run `helm lint` and `helm template` and assert on the rendered
manifests — but those are gated behind binary availability so the suite passes
with no cluster and no helm installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_DIR = REPO_ROOT / "deploy" / "helm" / "mnemozine"
VALUES_FILE = CHART_DIR / "values.yaml"
CHART_FILE = CHART_DIR / "Chart.yaml"
TEMPLATES_DIR = CHART_DIR / "templates"

HELM = shutil.which("helm")


def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# Static chart structure
# --------------------------------------------------------------------------- #


def test_chart_yaml_valid() -> None:
    chart = _load_yaml(CHART_FILE)
    assert chart["apiVersion"] == "v2"
    assert chart["name"] == "mnemozine"
    assert "version" in chart
    assert "appVersion" in chart


def test_required_templates_present() -> None:
    expected = {
        "_helpers.tpl",
        "configmap.yaml",
        "secret.yaml",
        "serviceaccount.yaml",
        "falkordb.yaml",
        "ollama.yaml",
        "qwen.yaml",
        "litellm.yaml",
        "mcp.yaml",
        "ingest.yaml",
        "maintenance.yaml",
    }
    present = {p.name for p in TEMPLATES_DIR.iterdir()}
    missing = expected - present
    assert not missing, f"missing chart templates: {sorted(missing)}"


def test_values_expose_images_endpoints_and_tuning() -> None:
    values = _load_yaml(VALUES_FILE)
    # Images parameterized.
    assert values["image"]["repository"]
    for dep in ("falkordb", "ollama", "qwen", "litellm"):
        assert "image" in values[dep], f"{dep} image not parameterized"
    # Endpoint overrides for external backends.
    assert "external" in values["endpoints"]
    for k in ("falkordbUrl", "ollamaBaseUrl", "extractionBaseUrl"):
        assert k in values["endpoints"]["external"], f"endpoints.external.{k} missing"


def test_values_cover_all_section_6_6_tuning_params() -> None:
    """PRD §6.6 tuning params must all be parameterized in values.tuning."""
    values = _load_yaml(VALUES_FILE)
    t = values["tuning"]
    assert t["inject"]["tokenBudget"] == 500
    assert t["crossref"]["relevanceThreshold"] == 0.8
    assert t["crossref"]["maxSuggestions"] == 2
    assert "vectorFallbackThreshold" in t["crossref"]
    m = t["maintenance"]
    for key in (
        "dedupEquivalenceThreshold",
        "edgeWeightFloor",
        "maxNodeDegree",
        "contradictionCandidateCap",
        "decayHalfLifeDays",
        "decayArchiveAfterDays",
        "cron",
    ):
        assert key in m, f"tuning.maintenance.{key} missing"
    assert "chunkMaxChars" in t["ingest"]
    assert "p95LatencyTargetMs" in t["retrieval"]
    assert "neighborhoodHops" in t["retrieval"]


def test_values_expose_knn_overfetch_and_ingest_source_flags() -> None:
    """New F2 knobs: KNN over-fetch tuning + ingest source enablement flags."""
    values = _load_yaml(VALUES_FILE)
    t = values["tuning"]
    # FR-RET-2 KNN over-fetch tuning.
    assert t["retrieval"]["knnOverfetchFactor"] == 10
    assert t["retrieval"]["knnOverfetchCap"] == 512
    # FR-ING-2/3/4 source enablement flags (Claude default-on, others off).
    ing = t["ingest"]
    assert ing["enableClaudeCode"] is True
    assert ing["enableGateway"] is False
    assert ing["enableHermes"] is False
    # Gateway / Hermes connection settings.
    for key in (
        "gatewayDefaultProject",
        "gatewayQueueMax",
        "hermesBaseUrl",
        "hermesDefaultProject",
        "hermesQueueMax",
        "hermesApiKey",
    ):
        assert key in ing, f"tuning.ingest.{key} missing"


def test_configmap_template_wires_new_knobs() -> None:
    """Static check: the ConfigMap template references every new env var."""
    text = (TEMPLATES_DIR / "configmap.yaml").read_text()
    for env in (
        "MNEMOZINE_RETRIEVAL__KNN_OVERFETCH_FACTOR",
        "MNEMOZINE_RETRIEVAL__KNN_OVERFETCH_CAP",
        "MNEMOZINE_INGEST__ENABLE_CLAUDE_CODE",
        "MNEMOZINE_INGEST__ENABLE_GATEWAY",
        "MNEMOZINE_INGEST__ENABLE_HERMES",
        "MNEMOZINE_INGEST__GATEWAY_DEFAULT_PROJECT",
        "MNEMOZINE_INGEST__GATEWAY_QUEUE_MAX",
        "MNEMOZINE_INGEST__HERMES_BASE_URL",
        "MNEMOZINE_INGEST__HERMES_DEFAULT_PROJECT",
        "MNEMOZINE_INGEST__HERMES_QUEUE_MAX",
    ):
        assert env in text, f"{env} not wired into ConfigMap template"
    # The Hermes API key is a credential -> Secret, NOT the ConfigMap.
    assert "MNEMOZINE_INGEST__HERMES_API_KEY" not in text


def test_secret_template_wires_hermes_api_key() -> None:
    text = (TEMPLATES_DIR / "secret.yaml").read_text()
    assert "MNEMOZINE_INGEST__HERMES_API_KEY" in text


def test_falkordb_has_persistence_values() -> None:
    values = _load_yaml(VALUES_FILE)
    persistence = values["falkordb"]["persistence"]
    assert persistence["enabled"] is True
    assert persistence["size"]


def test_all_workloads_have_resource_defaults() -> None:
    values = _load_yaml(VALUES_FILE)
    for comp in ("mcp", "ingest", "maintenance", "falkordb", "ollama", "qwen", "litellm"):
        assert "resources" in values[comp], f"{comp} has no resource defaults"
        res = values[comp]["resources"]
        assert "requests" in res and "limits" in res, f"{comp} missing requests/limits"


# --------------------------------------------------------------------------- #
# Live helm lint / template (skipped when helm is unavailable)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(HELM is None, reason="helm binary not available")
def test_helm_lint_passes() -> None:
    proc = subprocess.run(
        [HELM, "lint", str(CHART_DIR)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"helm lint failed:\n{proc.stdout}\n{proc.stderr}"


def _helm_template(*set_args: str) -> str:
    cmd = [HELM, "template", "mz", str(CHART_DIR)]
    for s in set_args:
        cmd += ["--set", s]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"helm template failed:\n{proc.stderr}"
    return proc.stdout


@pytest.mark.skipif(HELM is None, reason="helm binary not available")
def test_helm_template_default_renders_all_workloads() -> None:
    out = _helm_template()
    docs = [d for d in yaml.safe_load_all(out) if d]
    kinds: dict[str, int] = {}
    for d in docs:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    # FalkorDB is a StatefulSet (persistence); the rest are Deployments.
    assert kinds.get("StatefulSet", 0) == 1, "FalkorDB should be a StatefulSet"
    assert kinds.get("Deployment", 0) >= 4, "expected the mnemozine + dependency Deployments"
    assert kinds.get("ConfigMap", 0) >= 1
    assert kinds.get("Secret", 0) >= 1
    assert kinds.get("Service", 0) >= 4


@pytest.mark.skipif(HELM is None, reason="helm binary not available")
def test_helm_template_wires_tuning_into_configmap() -> None:
    out = _helm_template()
    cfg = None
    for d in yaml.safe_load_all(out):
        if d and d.get("kind") == "ConfigMap" and d["metadata"]["name"].endswith("-config"):
            if "MNEMOZINE_INJECT__TOKEN_BUDGET" in d.get("data", {}):
                cfg = d
                break
    assert cfg is not None, "shared mnemozine ConfigMap not rendered"
    data = cfg["data"]
    assert data["MNEMOZINE_INJECT__TOKEN_BUDGET"] == "500"
    assert data["MNEMOZINE_CROSSREF__RELEVANCE_THRESHOLD"] == "0.8"
    # Endpoint resolves to the in-cluster FalkorDB Service name.
    assert "falkordb" in data["MNEMOZINE_FALKORDB__URL"]


@pytest.mark.skipif(HELM is None, reason="helm binary not available")
def test_helm_template_renders_knn_and_ingest_flags() -> None:
    out = _helm_template()
    for d in yaml.safe_load_all(out):
        if (
            d
            and d.get("kind") == "ConfigMap"
            and "MNEMOZINE_RETRIEVAL__KNN_OVERFETCH_FACTOR" in d.get("data", {})
        ):
            data = d["data"]
            assert data["MNEMOZINE_RETRIEVAL__KNN_OVERFETCH_FACTOR"] == "10"
            assert data["MNEMOZINE_RETRIEVAL__KNN_OVERFETCH_CAP"] == "512"
            assert data["MNEMOZINE_INGEST__ENABLE_CLAUDE_CODE"] == "true"
            assert data["MNEMOZINE_INGEST__ENABLE_GATEWAY"] == "false"
            assert data["MNEMOZINE_INGEST__ENABLE_HERMES"] == "false"
            assert "MNEMOZINE_INGEST__HERMES_API_KEY" not in data
            return
    pytest.fail("KNN/ingest-flag env not rendered into the ConfigMap")


@pytest.mark.skipif(HELM is None, reason="helm binary not available")
def test_helm_template_overrides_knn_overfetch_factor() -> None:
    out = _helm_template("tuning.retrieval.knnOverfetchFactor=40")
    key = "MNEMOZINE_RETRIEVAL__KNN_OVERFETCH_FACTOR"
    for d in yaml.safe_load_all(out):
        if d and d.get("kind") == "ConfigMap" and key in d.get("data", {}):
            assert d["data"][key] == "40"
            return
    pytest.fail("knnOverfetchFactor override did not propagate")


@pytest.mark.skipif(HELM is None, reason="helm binary not available")
def test_helm_template_hermes_api_key_in_secret() -> None:
    out = _helm_template()
    for d in yaml.safe_load_all(out):
        if d and d.get("kind") == "Secret":
            sd = d.get("stringData") or {}
            if "MNEMOZINE_INGEST__HERMES_API_KEY" in sd:
                return
    pytest.fail("Hermes API key not rendered into the chart Secret")


@pytest.mark.skipif(HELM is None, reason="helm binary not available")
def test_helm_template_override_tuning_param() -> None:
    out = _helm_template("tuning.crossref.relevanceThreshold=0.95")
    key = "MNEMOZINE_CROSSREF__RELEVANCE_THRESHOLD"
    for d in yaml.safe_load_all(out):
        if d and d.get("kind") == "ConfigMap" and key in d.get("data", {}):
            assert d["data"][key] == "0.95"
            return
    pytest.fail("override did not propagate into the ConfigMap")


@pytest.mark.skipif(HELM is None, reason="helm binary not available")
def test_helm_template_external_endpoints() -> None:
    """With bundled backends disabled, external endpoint overrides are honoured."""
    out = _helm_template(
        "falkordb.enabled=false",
        "endpoints.external.falkordbUrl=redis://ext:6379",
        "ollama.enabled=false",
        "endpoints.external.ollamaBaseUrl=http://ext-ollama:11434",
        "litellm.enabled=false",
        "qwen.enabled=false",
        "endpoints.external.extractionBaseUrl=https://api.openai.com/v1",
    )
    for d in yaml.safe_load_all(out):
        if d and d.get("kind") == "ConfigMap" and "MNEMOZINE_FALKORDB__URL" in d.get("data", {}):
            data = d["data"]
            assert data["MNEMOZINE_FALKORDB__URL"] == "redis://ext:6379"
            assert data["MNEMOZINE_EMBEDDING__BASE_URL"] == "http://ext-ollama:11434"
            assert data["MNEMOZINE_EXTRACTION__BASE_URL"] == "https://api.openai.com/v1"
            return
    pytest.fail("external endpoints not rendered")


@pytest.mark.skipif(HELM is None, reason="helm binary not available")
def test_helm_template_maintenance_cronjob_variant() -> None:
    out = _helm_template("maintenance.asCronJob=true")
    kinds = {d.get("kind") for d in yaml.safe_load_all(out) if d}
    assert "CronJob" in kinds, "asCronJob=true should render a CronJob"
