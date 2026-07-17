import json
from datetime import datetime, timezone

from typer.testing import CliRunner

from conformguard.cli.main import app
from conformguard.storage.calibration_store import (
    CalibrationExample,
    CalibrationStore,
    LabelingSource,
)

runner = CliRunner()


def _build_store(path, n_good=150, n_bad=10):
    store = CalibrationStore(path)
    now = datetime.now(timezone.utc)
    for i in range(n_good):
        store.add(
            CalibrationExample(
                tool_name="search",
                context_bucket="prod",
                score=float(i) / n_good,
                outcome=True,
                labeling_source=LabelingSource.DETERMINISTIC,
                timestamp=now,
                calibration_set_version="v1",
            )
        )
    for i in range(n_bad):
        store.add(
            CalibrationExample(
                tool_name="search",
                context_bucket="prod",
                score=100.0 + i,
                outcome=False,
                labeling_source=LabelingSource.DETERMINISTIC,
                timestamp=now,
                calibration_set_version="v1",
            )
        )
    store.close()


class TestInspect:
    def test_reports_counts_and_range(self, tmp_path):
        db_path = tmp_path / "cal.db"
        _build_store(db_path)
        result = runner.invoke(app, ["inspect", "--store", str(db_path)])
        assert result.exit_code == 0
        assert "total examples: 160" in result.output
        assert "good=150" in result.output
        assert "bad=10" in result.output

    def test_missing_store_exits_nonzero(self, tmp_path):
        result = runner.invoke(app, ["inspect", "--store", str(tmp_path / "missing.db")])
        assert result.exit_code == 1

    def test_size_warning_shown_below_recommended_minimum(self, tmp_path):
        db_path = tmp_path / "cal.db"
        _build_store(db_path, n_good=50)
        result = runner.invoke(app, ["inspect", "--store", str(db_path), "--recommended-minimum", "1000"])
        assert "WARNING" in result.output


class TestThreshold:
    def test_computes_q_hat(self, tmp_path):
        db_path = tmp_path / "cal.db"
        _build_store(db_path, n_good=150)
        result = runner.invoke(app, ["threshold", "--alpha", "0.1", "--store", str(db_path)])
        assert result.exit_code == 0
        assert "q_hat=" in result.output
        assert "n=150" in result.output

    def test_no_matching_examples_exits_nonzero(self, tmp_path):
        db_path = tmp_path / "cal.db"
        _build_store(db_path, n_good=150)
        result = runner.invoke(
            app, ["threshold", "--alpha", "0.1", "--store", str(db_path), "--tool-name", "nonexistent"]
        )
        assert result.exit_code == 1


class TestCoverageCheck:
    def test_runs_and_reports_within_band(self, tmp_path):
        db_path = tmp_path / "cal.db"
        _build_store(db_path, n_good=500)
        result = runner.invoke(
            app,
            [
                "coverage-check",
                "--alpha",
                "0.1",
                "--calibration-size",
                "300",
                "--store",
                str(db_path),
                "--seed",
                "0",
            ],
        )
        assert result.exit_code == 0
        assert "mean observed coverage" in result.output
        assert "theoretical band" in result.output

    def test_insufficient_pool_exits_nonzero(self, tmp_path):
        db_path = tmp_path / "cal.db"
        _build_store(db_path, n_good=10)
        result = runner.invoke(
            app,
            ["coverage-check", "--alpha", "0.1", "--calibration-size", "50", "--store", str(db_path)],
        )
        assert result.exit_code == 1
