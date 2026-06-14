"""``mnemozine-eval`` console script — the EVAL harness CLI (PRD §9, deliverable #7).

Typer app exposing the harness offline against the committed gold-set fixture
(and the packaged in-memory fake store), so it runs end-to-end with no
FalkorDB/Ollama/Qwen. Commands:

* ``run``               — seed the gold set (optionally inflated) and run every
                          §9 metric once; exits non-zero on any failure.
* ``scaling``           — the headline §9 assertion: injection precision at 1x /
                          10x / 100x; exits non-zero if precision declines.
* ``knn-bench``         — KNN over-fetch recall@k benchmark (FR-RET-2): does the
                          configured ``retrieval.knn_overfetch_factor`` return the
                          true in-scope top_k for a low-selectivity scope inside a
                          large store at 1x / 10x / 100x? Exits non-zero on a drop.
* ``bootstrap-propose`` — auto-propose extracted candidates from a backlog and
                          write the operator review sheet (PRD §9 USER-TASK).
* ``bootstrap-finish``  — fold the operator-labeled review sheet into a gold set.
* ``show-gold``         — print a summary of a gold set.

The integration pass can later inject the *real* Retriever/CrossReferencer/
Extractor/StorageBackend; by default this CLI uses the self-contained offline
runner so the installed script is immediately useful.

Options use the ``Annotated[...]`` Typer idiom so no function call sits in an
argument default (ruff B008-clean).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from mnemozine.config import RetrievalSettings
from mnemozine.evals.bootstrap import (
    candidates_to_gold_set,
    propose_candidates,
    read_review_markdown,
    review_stats,
    write_review_markdown,
)
from mnemozine.evals.distractors import DEFAULT_INFLATION_LEVELS
from mnemozine.evals.goldset import (
    DEFAULT_GOLD_SET_PATH,
    GoldSet,
    load_gold_set,
    save_gold_set,
)
from mnemozine.evals.knn_bench import KnnBenchConfig, run_knn_overfetch_bench
from mnemozine.evals.runner import default_inmemory_runner

app = typer.Typer(
    help="Mnemozine eval harness (PRD §9): metrics, scaling, and eval-set bootstrap.",
    add_completion=False,
)

# Reusable option type aliases (Annotated keeps the call out of the default).
GoldOpt = Annotated[
    Path | None,
    typer.Option("--gold", help="Path to a gold-set JSON (defaults to the committed fixture)."),
]


def _load(gold_path: Path | None) -> GoldSet:
    return load_gold_set(gold_path) if gold_path else load_gold_set()


@app.command()
def run(
    gold_path: GoldOpt = None,
    inflation: Annotated[
        int,
        typer.Option(
            "--inflation", "-x", help="Distractor multiplier per gold memory (1x/10x/100x)."
        ),
    ] = 1,
) -> None:
    """Run every §9 metric once against the gold set (offline)."""

    runner = default_inmemory_runner(gold_set=_load(gold_path))
    report = asyncio.run(runner.run_all(inflation_multiplier=inflation))
    typer.echo(report.render())
    raise typer.Exit(code=0 if report.passed else 1)


@app.command()
def scaling(
    gold_path: GoldOpt = None,
    levels: Annotated[
        str,
        typer.Option("--levels", help="Comma-separated inflation levels (default 1,10,100)."),
    ] = ",".join(str(x) for x in DEFAULT_INFLATION_LEVELS),
    tolerance: Annotated[
        float,
        typer.Option("--tolerance", help="Allowed precision drop below baseline before failing."),
    ] = 0.0,
) -> None:
    """Precision-stays-flat assertion across inflation levels (PRD §9 headline)."""

    parsed_levels = tuple(int(x) for x in levels.split(",") if x.strip())
    runner = default_inmemory_runner(gold_set=_load(gold_path))
    report = asyncio.run(runner.precision_scaling(levels=parsed_levels, tolerance=tolerance))
    typer.echo(report.render())
    raise typer.Exit(code=0 if report.passed else 1)


@app.command("knn-bench")
def knn_bench(
    gold_path: GoldOpt = None,
    factor: Annotated[
        int | None,
        typer.Option(
            "--factor",
            help="Override retrieval.knn_overfetch_factor (default: config value, 10).",
        ),
    ] = None,
    cap: Annotated[
        int | None,
        typer.Option(
            "--cap",
            help="Override retrieval.knn_overfetch_cap (default: config value, 512).",
        ),
    ] = None,
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Retrieval depth recall@k is measured at."),
    ] = 10,
    in_scope_fraction: Annotated[
        float | None,
        typer.Option(
            "--in-scope-fraction",
            help=(
                "Fraction of the store the scope holds (lower = more selective). "
                "A scope needs factor >= 1/fraction; e.g. 0.05 starves factor=10."
            ),
        ),
    ] = None,
    levels: Annotated[
        str,
        typer.Option("--levels", help="Comma-separated inflation levels (default 1,10,100)."),
    ] = ",".join(str(x) for x in DEFAULT_INFLATION_LEVELS),
    recall_floor: Annotated[
        float,
        typer.Option(
            "--recall-floor",
            help="Minimum recall@k each level must hold (1.0 = no in-scope miss).",
        ),
    ] = 1.0,
) -> None:
    """KNN over-fetch recall@k benchmark across 1x/10x/100x (FR-RET-2, PRD §9).

    Measures whether the configured ``retrieval.knn_overfetch_factor`` returns
    the true in-scope top_k (recall@k vs an exhaustive in-process baseline) for a
    low-selectivity scope inside a large global store. The recommended tuning
    rule (``factor >= 1 / in_scope_fraction``) is documented in the benchmark's
    output and module docstring. Exits non-zero if recall@k drops below the floor.
    """

    retrieval = RetrievalSettings()
    if factor is not None:
        retrieval.knn_overfetch_factor = factor
    if cap is not None:
        retrieval.knn_overfetch_cap = cap
    parsed_levels = tuple(int(x) for x in levels.split(",") if x.strip())
    bench_config = (
        KnnBenchConfig(top_k=top_k, in_scope_fraction_target=in_scope_fraction)
        if in_scope_fraction is not None
        else KnnBenchConfig(top_k=top_k)
    )
    report = run_knn_overfetch_bench(
        retrieval=retrieval,
        gold_set=_load(gold_path),
        config=bench_config,
        levels=parsed_levels,
        recall_floor=recall_floor,
    )
    typer.echo(report.render())
    raise typer.Exit(code=0 if report.passed else 1)


@app.command("show-gold")
def show_gold(gold_path: GoldOpt = None) -> None:
    """Print a summary of a gold set."""

    gs = _load(gold_path)
    typer.echo(f"gold set: {gs.name}")
    typer.echo(f"  memories:        {len(gs.memories)}")
    typer.echo(f"  injection cases: {len(gs.injection_cases)}")
    typer.echo(f"  preference cases:{len(gs.preference_cases)}")
    typer.echo(f"  crossref cases:  {len(gs.crossref_cases)}")
    typer.echo(f"  classifier cases:{len(gs.classifier_cases)}")
    typer.echo(f"  no-leak cases:   {len(gs.no_leak_cases)}")


@app.command("bootstrap-propose")
def bootstrap_propose(
    review_out: Annotated[
        Path,
        typer.Option("--out", help="Where to write the operator review Markdown sheet."),
    ] = Path("eval_review.md"),
) -> None:
    """Auto-propose eval candidates and write the operator review sheet (PRD §9).

    This is the **operator deliverable** half of §9: it extracts candidates from
    the historical backlog and writes a Markdown sheet for the operator to label
    yes/no. Offline it uses a tiny demo backlog + the harness extractor so the
    command is exercisable without a live ingest source; the integration pass can
    point it at the real ``IngestSource.backfill`` + ``Extractor``.
    """

    from datetime import UTC, datetime

    from mnemozine.evals.bootstrap import Candidate
    from mnemozine.evals.harness_adapters import KeywordExtractor
    from mnemozine.interfaces import RetrievalContext
    from mnemozine.schema.events import IngestEvent, Role, Source
    from mnemozine.schema.models import MemoryType

    base = datetime(2026, 6, 13, tzinfo=UTC)
    demo_chunk = [
        IngestEvent(
            source=Source.CLAUDE_CODE,
            project="demo",
            session_id="backlog-1",
            timestamp=base,
            role=Role.USER,
            content="I prefer thiserror over anyhow in Rust.",
        ),
    ]

    # KeywordExtractor.extract is intentionally not implemented (the harness only
    # needs classify); for the offline demo we build a candidate from classify.
    async def _demo() -> None:
        extractor = KeywordExtractor()
        candidates = []
        for i, ev in enumerate(demo_chunk):
            cls = await extractor.classify(ev.content, RetrievalContext(project=ev.project))
            candidates.append(
                Candidate(
                    candidate_id=f"cand-{i:04d}",
                    content=ev.content,
                    proposed_type=MemoryType.from_split(
                        cls.scope_decision, cls.cross_ref_candidate
                    ),
                    scope=cls.scope.as_str(),
                    entities=cls.entities,
                    confidence=cls.confidence,
                    source_session=ev.session_id,
                )
            )
        out = write_review_markdown(candidates, review_out)
        typer.echo(f"wrote {len(candidates)} candidate(s) to {out}")
        typer.echo("Edit the file (tick `- [x] keep`), then run bootstrap-finish.")

    asyncio.run(_demo())


@app.command("bootstrap-finish")
def bootstrap_finish(
    review_in: Annotated[
        Path, typer.Option("--in", help="The operator-edited review sheet.")
    ] = Path("eval_review.md"),
    gold_out: Annotated[
        Path, typer.Option("--out", help="Where to write the resulting gold-set JSON.")
    ] = DEFAULT_GOLD_SET_PATH,
) -> None:
    """Fold an operator-labeled review sheet into a committed gold set (PRD §9)."""

    candidates = read_review_markdown(review_in)
    stats = review_stats(candidates)
    gs = candidates_to_gold_set(candidates)
    save_gold_set(gs, gold_out)
    typer.echo(
        f"reviewed={stats['total']} kept={stats['keep']} dropped={stats['drop']} "
        f"-> wrote gold set ({len(gs.memories)} memories) to {gold_out}"
    )


# Re-export the propose function so callers/tests can drive a real backlog.
__all__ = ["app", "main", "propose_candidates"]


def main() -> None:
    """Console-script entrypoint for ``mnemozine-eval`` (see integration notes)."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()
