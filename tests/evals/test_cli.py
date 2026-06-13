"""Unit tests for the ``mnemozine-eval`` Typer CLI (PRD §9, deliverable #7)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mnemozine.evals.cli import app

runner = CliRunner()


def test_show_gold() -> None:
    result = runner.invoke(app, ["show-gold"])
    assert result.exit_code == 0
    assert "gold set:" in result.stdout
    assert "memories:" in result.stdout


def test_run_passes_on_fixture() -> None:
    result = runner.invoke(app, ["run", "--inflation", "1"])
    assert result.exit_code == 0
    assert "overall: PASS" in result.stdout
    # Each §9 metric is reported.
    assert "injection_precision_at_k" in result.stdout
    assert "no_leak_check" in result.stdout


def test_run_under_inflation() -> None:
    result = runner.invoke(app, ["run", "-x", "10"])
    assert result.exit_code == 0
    assert "inflation 10x" in result.stdout


def test_scaling_command() -> None:
    result = runner.invoke(app, ["scaling", "--levels", "1,10,100"])
    assert result.exit_code == 0
    assert "Precision-stays-flat" in result.stdout
    assert "PASS" in result.stdout


def test_bootstrap_propose_then_finish(tmp_path: Path) -> None:
    review = tmp_path / "review.md"
    gold = tmp_path / "gold.json"

    # propose
    r1 = runner.invoke(app, ["bootstrap-propose", "--out", str(review)])
    assert r1.exit_code == 0
    assert review.exists()
    text = review.read_text()
    assert "- [ ] keep" in text

    # operator ticks keep
    review.write_text(text.replace("- [ ] keep", "- [x] keep"))

    # finish
    r2 = runner.invoke(
        app, ["bootstrap-finish", "--in", str(review), "--out", str(gold)]
    )
    assert r2.exit_code == 0
    assert gold.exists()
    assert "kept=1" in r2.stdout

    # the produced gold set is loadable + runnable
    r3 = runner.invoke(app, ["show-gold", "--gold", str(gold)])
    assert r3.exit_code == 0
    assert "classifier cases:1" in r3.stdout


def test_run_with_custom_gold(tmp_path: Path) -> None:
    # bootstrap a gold set, then run against it.
    review = tmp_path / "review.md"
    gold = tmp_path / "gold.json"
    runner.invoke(app, ["bootstrap-propose", "--out", str(review)])
    review.write_text(review.read_text().replace("- [ ] keep", "- [x] keep"))
    runner.invoke(app, ["bootstrap-finish", "--in", str(review), "--out", str(gold)])
    result = runner.invoke(app, ["run", "--gold", str(gold)])
    assert result.exit_code == 0
    # Only the classifier metric has cases; it should pass on the kept pref.
    assert "classifier_accuracy" in result.stdout
