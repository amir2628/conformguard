"""conformguard CLI: inspect calibration data, thresholds, and run coverage checks."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    import numpy as np

from conformguard.core.quantile import conformal_quantile
from conformguard.storage.calibration_store import (
    DEFAULT_STALENESS_THRESHOLD,
    RECOMMENDED_MINIMUM_SIZE,
    CalibrationStore,
)
from conformguard.validation.coverage_check import run_coverage_validation
from conformguard.validation.multi_check_comparison import run_multi_check_comparison

app = typer.Typer(
    name="conformguard",
    help="Inspect calibration data, thresholds, and run coverage checks for conformguard.",
    no_args_is_help=True,
)


def _open_store(store: Path) -> CalibrationStore:
    if not store.exists():
        typer.secho(f"no calibration store found at {store}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    return CalibrationStore(store)


@app.command()
def inspect(
    store: Path = typer.Option(Path(".conformguard/calibration.db"), help="Path to the calibration store."),
    tool_name: str | None = typer.Option(None, help="Filter by tool name."),
    context_bucket: str | None = typer.Option(None, help="Filter by context bucket."),
    calibration_set_version: str | None = typer.Option(None, help="Filter by calibration set version."),
    staleness_days: int = typer.Option(
        DEFAULT_STALENESS_THRESHOLD.days, help="Staleness warning threshold, in days."
    ),
    recommended_minimum: int = typer.Option(
        RECOMMENDED_MINIMUM_SIZE, help="Recommended minimum calibration set size for the size warning."
    ),
) -> None:
    """Inspect calibration data: counts, good/bad breakdown, timestamp range, and warnings."""
    db = _open_store(store)
    filters: dict[str, object] = {}
    if tool_name is not None:
        filters["tool_name"] = tool_name
    if context_bucket is not None:
        filters["context_bucket"] = context_bucket
    if calibration_set_version is not None:
        filters["calibration_set_version"] = calibration_set_version

    total = db.count(**filters)
    good = db.count(outcome=True, **filters)
    bad = db.count(outcome=False, **filters)
    time_range = db.timestamp_range(**filters)

    typer.echo(f"calibration store: {store}")
    typer.echo(f"filters: {filters or '(none)'}")
    typer.echo(f"total examples: {total}  (good={good}, bad={bad})")
    if time_range:
        oldest, newest = time_range
        typer.echo(f"timestamp range: {oldest.isoformat()} -> {newest.isoformat()}")
    else:
        typer.echo("timestamp range: (no examples)")

    staleness_warning = db.staleness_warning(max_age=timedelta(days=staleness_days), **filters)
    if staleness_warning:
        typer.secho(f"WARNING: {staleness_warning}", fg=typer.colors.YELLOW)

    size_warning = db.size_warning(recommended_minimum=recommended_minimum, outcome=True, **filters)
    if size_warning:
        typer.secho(f"WARNING: {size_warning}", fg=typer.colors.YELLOW)

    db.close()


@app.command()
def threshold(
    alpha: float = typer.Option(..., help="Target miscoverage rate, in (0, 1)."),
    store: Path = typer.Option(Path(".conformguard/calibration.db"), help="Path to the calibration store."),
    tool_name: str | None = typer.Option(None, help="Filter by tool name."),
    context_bucket: str | None = typer.Option(None, help="Filter by context bucket."),
    calibration_set_version: str | None = typer.Option(None, help="Filter by calibration set version."),
) -> None:
    """Compute and print q_hat directly from a calibration store's good-outcome scores."""
    db = _open_store(store)
    filters: dict[str, object] = {"outcome": True}
    if tool_name is not None:
        filters["tool_name"] = tool_name
    if context_bucket is not None:
        filters["context_bucket"] = context_bucket
    if calibration_set_version is not None:
        filters["calibration_set_version"] = calibration_set_version

    examples = db.query(**filters)
    db.close()

    if not examples:
        typer.secho("no good-outcome examples match the given filters", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    scores = [example.score for example in examples]
    q_hat = conformal_quantile(scores, alpha=alpha)
    typer.echo(f"n={len(scores)} alpha={alpha} q_hat={q_hat}")


@app.command(name="coverage-check")
def coverage_check(
    alpha: float = typer.Option(..., help="Target miscoverage rate, in (0, 1)."),
    calibration_size: int = typer.Option(..., help="Calibration set size to use for each repeated split."),
    store: Path = typer.Option(Path(".conformguard/calibration.db"), help="Path to the calibration store."),
    tool_name: str | None = typer.Option(None, help="Filter by tool name."),
    context_bucket: str | None = typer.Option(None, help="Filter by context bucket."),
    trials: int = typer.Option(100, help="Number of repeated calibration/test splits (R)."),
    seed: int | None = typer.Option(None, help="Random seed for reproducible splits."),
) -> None:
    """Run the empirical coverage validation suite against a calibration store's good-outcome scores."""
    db = _open_store(store)
    filters: dict[str, object] = {"outcome": True}
    if tool_name is not None:
        filters["tool_name"] = tool_name
    if context_bucket is not None:
        filters["context_bucket"] = context_bucket

    examples = db.query(**filters)
    db.close()

    pool = [example.score for example in examples]
    if len(pool) <= calibration_size:
        typer.secho(
            f"pool has only {len(pool)} good-outcome examples, which does not leave a "
            f"non-empty test set for calibration_size={calibration_size}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    result = run_coverage_validation(
        pool, alpha=alpha, calibration_size=calibration_size, n_trials=trials, seed=seed
    )
    typer.echo(f"pool size: {len(pool)}")
    typer.echo(f"trials: {result.n_trials}")
    typer.echo(f"mean observed coverage: {result.mean_observed_coverage:.4f}")
    typer.echo(f"theoretical band ({result.band.confidence:.0%} CI): [{result.band.low:.4f}, {result.band.high:.4f}]")
    typer.echo(f"theoretical mean: {result.band.mean:.4f}")
    status = "WITHIN BAND" if result.within_band else "OUTSIDE BAND"
    color = typer.colors.GREEN if result.within_band else typer.colors.RED
    typer.secho(status, fg=color, bold=True)


def _load_multi_check_data(path: Path) -> tuple[list[str], "np.ndarray", list[bool]]:
    """Load Phase 2 multi-check calibration data from a JSON file.

    Expected shape: a list of records, each
    ``{"scores": {"<check_name>": <float>, ...}, "outcome": <bool>}``.
    All records must declare the same set of check names. Returns
    (sorted check names, an (N, K) score matrix in that column order,
    outcome flags) -- this JSON format, not the SQLite calibration store,
    is Phase 2's calibration-data interchange for now: the store's schema
    is one score per row with no notion of "these K rows are simultaneous
    checks on the same call," and retrofitting that grouping is a
    real, separate piece of storage-layer work this CLI extension
    deliberately does not take on speculatively.
    """
    import numpy as np

    if not path.exists():
        typer.secho(f"no multi-check data file found at {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    records = json.loads(path.read_text())
    if not records:
        typer.secho(f"{path} contains no records", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    check_names = sorted(records[0]["scores"])
    for i, record in enumerate(records):
        if sorted(record["scores"]) != check_names:
            typer.secho(
                f"record {i} declares checks {sorted(record['scores'])}, expected {check_names} "
                f"(all records must declare the same set of check names)",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

    matrix = np.array([[record["scores"][name] for name in check_names] for record in records], dtype=float)
    outcomes = [bool(record["outcome"]) for record in records]
    return check_names, matrix, outcomes


@app.command(name="multi-check-threshold")
def multi_check_threshold(
    data: Path = typer.Option(..., help="Path to a JSON file of multi-check calibration records."),
    alpha: float = typer.Option(..., help="Target joint miscoverage rate, in (0, 1)."),
) -> None:
    """Compute the joint (max-score) threshold and print a per-check breakdown.

    The per-check breakdown reports each check's own marginal quantile at
    the same alpha (for context) and how often each check was the
    "binding" one -- i.e. the max -- among the good-outcome calibration
    examples, which is a diagnostic for which check is actually driving
    the joint threshold.
    """
    import numpy as np

    check_names, matrix, outcomes = _load_multi_check_data(data)
    good_matrix = matrix[np.array(outcomes)]
    if good_matrix.shape[0] == 0:
        typer.secho("no good-outcome (outcome=true) records in the data file", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    max_scores = good_matrix.max(axis=1).tolist()
    q_hat = conformal_quantile(max_scores, alpha=alpha)

    typer.echo(f"n={good_matrix.shape[0]}  k={len(check_names)}  alpha={alpha}  q_hat(joint)={q_hat}")
    typer.echo()
    typer.echo("per-check breakdown:")
    binding_counts = np.argmax(good_matrix, axis=1)
    for j, name in enumerate(check_names):
        marginal_q = conformal_quantile(good_matrix[:, j].tolist(), alpha=alpha)
        binding_fraction = float(np.mean(binding_counts == j))
        typer.echo(
            f"  {name!r}: min={good_matrix[:, j].min():.4f} max={good_matrix[:, j].max():.4f} "
            f"mean={good_matrix[:, j].mean():.4f}  own-marginal-q_hat(alpha={alpha})={marginal_q:.4f}  "
            f"was-the-max-in={binding_fraction:.1%} of examples"
        )


@app.command(name="multi-check-coverage-check")
def multi_check_coverage_check(
    data: Path = typer.Option(..., help="Path to a JSON file of multi-check calibration records."),
    alpha: float = typer.Option(..., help="Target joint miscoverage rate, in (0, 1)."),
    calibration_size: int = typer.Option(..., help="Calibration set size to use for each repeated split."),
    trials: int = typer.Option(100, help="Number of repeated calibration/test splits (R)."),
    seed: int | None = typer.Option(None, help="Random seed for reproducible splits."),
    compare: bool = typer.Option(
        False,
        help="Also run naive-independent and Bonferroni for comparison (PROJECT_SPEC §3 Phase 2). "
        "NOTE: --compare currently reports only good-call coverage, not the bad-rejection "
        "efficiency metric (validation/multi_check_comparison.py's other metric) -- there is no "
        "CLI-level way yet to supply a separate bad/anomalous pool.",
    ),
) -> None:
    """Run the empirical coverage validation suite for joint multi-check calibration."""
    import numpy as np

    check_names, matrix, outcomes = _load_multi_check_data(data)
    good_matrix = matrix[np.array(outcomes)]
    if good_matrix.shape[0] <= calibration_size:
        typer.secho(
            f"pool has only {good_matrix.shape[0]} good-outcome examples, which does not leave a "
            f"non-empty test set for calibration_size={calibration_size}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    if not compare:
        from conformguard.validation.coverage_check import run_coverage_validation as _run

        max_scores = good_matrix.max(axis=1).tolist()
        result = _run(max_scores, alpha=alpha, calibration_size=calibration_size, n_trials=trials, seed=seed)
        typer.echo(f"pool size: {good_matrix.shape[0]}  k={len(check_names)}  trials: {result.n_trials}")
        typer.echo(f"mean observed joint coverage: {result.mean_observed_coverage:.4f}")
        typer.echo(f"theoretical band ({result.band.confidence:.0%} CI): [{result.band.low:.4f}, {result.band.high:.4f}]")
        status = "WITHIN BAND" if result.within_band else "OUTSIDE BAND"
        typer.secho(status, fg=(typer.colors.GREEN if result.within_band else typer.colors.RED), bold=True)
        return

    results = run_multi_check_comparison(
        good_matrix, alpha=alpha, calibration_size=calibration_size, n_trials=trials, seed=seed
    )
    typer.echo(f"pool size: {good_matrix.shape[0]}  k={len(check_names)}  trials: {trials}  alpha: {alpha}")
    typer.echo()
    for name in ("joint", "naive", "bonferroni"):
        result = results[name]
        typer.echo(f"{name}: mean good-call coverage={result.mean_good_coverage:.4f} (target={1 - alpha:.4f})")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
