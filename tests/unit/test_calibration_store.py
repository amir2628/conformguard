from datetime import datetime, timedelta, timezone

import pytest

from conformguard.storage.calibration_store import (
    CalibrationExample,
    CalibrationStore,
    LabelingSource,
)


def _example(
    tool_name: str = "search",
    context_bucket: str = "default",
    score: float = 0.5,
    outcome: bool = True,
    labeling_source: LabelingSource = LabelingSource.DETERMINISTIC,
    timestamp: datetime | None = None,
    calibration_set_version: str = "v1",
) -> CalibrationExample:
    return CalibrationExample(
        tool_name=tool_name,
        context_bucket=context_bucket,
        score=score,
        outcome=outcome,
        labeling_source=labeling_source,
        timestamp=timestamp or datetime.now(timezone.utc),
        calibration_set_version=calibration_set_version,
    )


@pytest.fixture
def store():
    s = CalibrationStore(":memory:")
    yield s
    s.close()


class TestReadWrite:
    def test_add_and_query_round_trips(self, store):
        example = _example(score=0.42)
        example_id = store.add(example)
        assert example_id > 0

        results = store.query(tool_name="search")
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.42)
        assert results[0].outcome is True
        assert results[0].labeling_source == LabelingSource.DETERMINISTIC

    def test_add_many(self, store):
        examples = [_example(score=float(i)) for i in range(5)]
        ids = store.add_many(examples)
        assert len(ids) == 5
        assert store.count() == 5

    def test_query_filters_by_tool_name(self, store):
        store.add(_example(tool_name="search"))
        store.add(_example(tool_name="write_file"))
        assert store.count(tool_name="search") == 1
        assert store.count(tool_name="write_file") == 1
        assert store.count() == 2

    def test_query_filters_by_context_bucket(self, store):
        store.add(_example(context_bucket="prod"))
        store.add(_example(context_bucket="staging"))
        assert store.count(context_bucket="prod") == 1

    def test_query_filters_by_outcome(self, store):
        store.add(_example(outcome=True))
        store.add(_example(outcome=False))
        assert store.count(outcome=True) == 1
        assert store.count(outcome=False) == 1

    def test_query_filters_by_calibration_set_version(self, store):
        store.add(_example(calibration_set_version="v1"))
        store.add(_example(calibration_set_version="v2"))
        assert store.count(calibration_set_version="v1") == 1

    def test_query_orders_by_timestamp_ascending(self, store):
        now = datetime.now(timezone.utc)
        store.add(_example(score=3, timestamp=now))
        store.add(_example(score=1, timestamp=now - timedelta(days=2)))
        store.add(_example(score=2, timestamp=now - timedelta(days=1)))
        results = store.query()
        assert [r.score for r in results] == [1, 2, 3]

    def test_empty_store_returns_empty_list(self, store):
        assert store.query() == []
        assert store.count() == 0


class TestStalenessWarning:
    def test_no_warning_when_fresh(self, store):
        store.add(_example(timestamp=datetime.now(timezone.utc)))
        assert store.staleness_warning() is None

    def test_warning_when_older_than_threshold(self, store):
        old = datetime.now(timezone.utc) - timedelta(days=200)
        store.add(_example(timestamp=old))
        warning = store.staleness_warning(max_age=timedelta(days=90))
        assert warning is not None
        assert "stale" in warning.lower()

    def test_no_warning_when_no_data(self, store):
        assert store.staleness_warning() is None

    def test_respects_filters(self, store):
        old = datetime.now(timezone.utc) - timedelta(days=200)
        fresh = datetime.now(timezone.utc)
        store.add(_example(tool_name="old_tool", timestamp=old))
        store.add(_example(tool_name="fresh_tool", timestamp=fresh))
        assert store.staleness_warning(tool_name="fresh_tool") is None
        assert store.staleness_warning(tool_name="old_tool") is not None


class TestSizeWarning:
    def test_warning_below_recommended_minimum(self, store):
        for i in range(5):
            store.add(_example(score=float(i)))
        warning = store.size_warning(recommended_minimum=1000)
        assert warning is not None
        assert "5" in warning

    def test_no_warning_at_or_above_recommended_minimum(self, store):
        for i in range(10):
            store.add(_example(score=float(i)))
        assert store.size_warning(recommended_minimum=10) is None

    def test_respects_filters(self, store):
        for i in range(10):
            store.add(_example(tool_name="popular", score=float(i)))
        store.add(_example(tool_name="rare", score=0.1))
        assert store.size_warning(recommended_minimum=5, tool_name="popular") is None
        assert store.size_warning(recommended_minimum=5, tool_name="rare") is not None


class TestTimestampRange:
    def test_returns_none_when_empty(self, store):
        assert store.timestamp_range() is None

    def test_returns_min_and_max(self, store):
        now = datetime.now(timezone.utc)
        store.add(_example(timestamp=now - timedelta(days=5)))
        store.add(_example(timestamp=now))
        oldest, newest = store.timestamp_range()
        assert newest - oldest == timedelta(days=5)


class TestPersistence:
    def test_reopening_same_file_preserves_data(self, tmp_path):
        db_path = tmp_path / "cal.db"
        s1 = CalibrationStore(db_path)
        s1.add(_example(score=0.7))
        s1.close()

        s2 = CalibrationStore(db_path)
        try:
            assert s2.count() == 1
            assert s2.query()[0].score == pytest.approx(0.7)
        finally:
            s2.close()
