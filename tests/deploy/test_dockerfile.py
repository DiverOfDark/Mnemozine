"""Static assertions on the Mnemozine Dockerfile (PRD Deliverable #1).

No image build is performed (that needs a Docker daemon + network); these are
text-level checks that the multi-stage build is shaped correctly and that the
image can serve every mnemozine console_script.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "deploy" / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"


def _dockerfile_text() -> str:
    return DOCKERFILE.read_text()


def test_dockerfile_exists() -> None:
    assert DOCKERFILE.is_file(), f"missing {DOCKERFILE}"


def test_dockerfile_is_multistage() -> None:
    text = _dockerfile_text().lower()
    from_lines = [ln for ln in text.splitlines() if ln.strip().startswith("from ")]
    assert len(from_lines) >= 2, "Dockerfile is not multi-stage (need >=2 FROM)"
    assert "as builder" in text, "no builder stage"
    assert "as runtime" in text, "no runtime stage"


def test_dockerfile_installs_project() -> None:
    text = _dockerfile_text()
    # Uses uv (the chosen installer) and installs the project (provides scripts).
    assert "uv" in text, "Dockerfile does not use uv"
    assert "pyproject.toml" in text, "Dockerfile does not copy pyproject metadata"


def test_dockerfile_default_cmd_is_a_console_script() -> None:
    text = _dockerfile_text()
    # The default command should be one of the real console_scripts.
    assert "mnemozine-mcp" in text, "default CMD is not a mnemozine console script"


def test_dockerfile_runs_as_non_root() -> None:
    text = _dockerfile_text()
    assert "USER mnemozine" in text or "useradd" in text, "image runs as root"


def test_dockerignore_excludes_secrets_and_venv() -> None:
    text = DOCKERIGNORE.read_text()
    for pat in (".env", ".venv", ".git"):
        assert pat in text, f".dockerignore should exclude {pat}"
    # But the example env is allowed back in for reference.
    assert "!.env.example" in text
