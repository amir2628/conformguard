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


def _write_multi_check_data(path, n=200, k_names=("check_a", "check_b"), bad_fraction=0.0, seed=0):
    import random as _random

    rng = _random.Random(seed)
    records = []
    n_bad = int(n * bad_fraction)
    for i in range(n):
        outcome = i >= n_bad
        records.append(
            {
                "scores": {name: rng.uniform(0.0, 1.0) for name in k_names},
                "outcome": outcome,
            }
        )
    path.write_text(json.dumps(records))


class TestMultiCheckThreshold:
    def test_computes_joint_threshold_and_breakdown(self, tmp_path):
        data_path = tmp_path / "multi_check.json"
        _write_multi_check_data(data_path, n=200)
        result = runner.invoke(app, ["multi-check-threshold", "--data", str(data_path), "--alpha", "0.1"])
        assert result.exit_code == 0
        assert "q_hat(joint)=" in result.output
        assert "check_a" in result.output
        assert "check_b" in result.output
        assert "was-the-max-in=" in result.output

    def test_missing_file_exits_nonzero(self, tmp_path):
        result = runner.invoke(
            app, ["multi-check-threshold", "--data", str(tmp_path / "missing.json"), "--alpha", "0.1"]
        )
        assert result.exit_code == 1

    def test_mismatched_check_names_exits_nonzero(self, tmp_path):
        data_path = tmp_path / "bad.json"
        data_path.write_text(
            json.dumps(
                [
                    {"scores": {"a": 0.1, "b": 0.2}, "outcome": True},
                    {"scores": {"a": 0.1, "c": 0.3}, "outcome": True},
                ]
            )
        )
        result = runner.invoke(app, ["multi-check-threshold", "--data", str(data_path), "--alpha", "0.1"])
        assert result.exit_code == 1

    def test_no_good_outcome_records_exits_nonzero(self, tmp_path):
        data_path = tmp_path / "all_bad.json"
        _write_multi_check_data(data_path, n=50, bad_fraction=1.0)
        result = runner.invoke(app, ["multi-check-threshold", "--data", str(data_path), "--alpha", "0.1"])
        assert result.exit_code == 1


class TestMultiCheckCoverageCheck:
    def test_joint_only_reports_within_band(self, tmp_path):
        data_path = tmp_path / "multi_check.json"
        _write_multi_check_data(data_path, n=2000)
        result = runner.invoke(
            app,
            [
                "multi-check-coverage-check",
                "--data",
                str(data_path),
                "--alpha",
                "0.1",
                "--calibration-size",
                "1000",
                "--seed",
                "0",
            ],
        )
        assert result.exit_code == 0
        assert "mean observed joint coverage" in result.output

    def test_compare_reports_all_three_methods(self, tmp_path):
        data_path = tmp_path / "multi_check.json"
        _write_multi_check_data(data_path, n=2000)
        result = runner.invoke(
            app,
            [
                "multi-check-coverage-check",
                "--data",
                str(data_path),
                "--alpha",
                "0.1",
                "--calibration-size",
                "1000",
                "--seed",
                "0",
                "--compare",
            ],
        )
        assert result.exit_code == 0
        assert "joint:" in result.output
        assert "naive:" in result.output
        assert "bonferroni:" in result.output

    def test_insufficient_pool_exits_nonzero(self, tmp_path):
        data_path = tmp_path / "small.json"
        _write_multi_check_data(data_path, n=10)
        result = runner.invoke(
            app,
            ["multi-check-coverage-check", "--data", str(data_path), "--alpha", "0.1", "--calibration-size", "50"],
        )
        assert result.exit_code == 1
