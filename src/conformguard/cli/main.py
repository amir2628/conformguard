"""conformguard CLI: inspect calibration data, thresholds, and run coverage checks."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import typer

from conformguard.core.quantile import conformal_quantile
from conformguard.storage.calibration_store import (
    DEFAULT_STALENESS_THRESHOLD,
    RECOMMENDED_MINIMUM_SIZE,
    CalibrationStore,
)
from conformguard.validation.coverage_check import run_coverage_validation

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
