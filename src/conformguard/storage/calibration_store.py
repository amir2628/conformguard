"""Local SQLite store for calibration examples.

No network dependency required to function, mirroring the local-first
philosophy of the calibration store described in the project spec. This
module only persists and queries records; it has no opinion on whether a
given set of records is *sufficient* to calibrate on -- that hard-minimum
decision belongs to core/calibration.py, since raising is a calibration-time
policy, not a storage concern. This module does surface staleness/size
*warnings*, since those are properties of the stored data itself.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

# Below this many examples, coverage fluctuation is wide enough that the
# calibration set should be treated as provisional; see Angelopoulos & Bates
# Table 1 (arXiv:2107.07511), which shows n ~= 1000 keeps coverage
# fluctuation within +/-2 percentage points at alpha=0.1. This is a
# *warning* threshold, not a hard floor -- see core/calibration.py for the
# hard floor below which calibrate() refuses to produce a threshold at all.
RECOMMENDED_MINIMUM_SIZE = 1000

# Exchangeability requires the calibration set to represent the *current*
# deployment distribution; a set older than this is more likely to have
# drifted. Configurable per call to any staleness-checking method.
DEFAULT_STALENESS_THRESHOLD = timedelta(days=90)


class LabelingSource(str, Enum):
    """How an outcome label was produced. Part of the guarantee statement.

    A guarantee calibrated on human-labeled outcomes should be presented
    differently from one calibrated on a noisy downstream-success proxy
    (see PROJECT_SPEC §4.3, §10) -- this is why the source travels with
    every record instead of being a global setting.
    """

    DETERMINISTIC = "deterministic"
    HUMAN = "human"
    DOWNSTREAM_PROXY = "downstream_proxy"


class CalibrationExample(BaseModel):
    """One (score, outcome) record with the metadata the guarantee depends on."""

    id: int | None = None
    tool_name: str
    context_bucket: str = "default"
    score: float
    outcome: bool
    labeling_source: LabelingSource
    timestamp: datetime
    calibration_set_version: str


class CalibrationStore:
    """SQLite-backed store for calibration examples.

    Opens (and creates, if missing) a database file at ``path``. Pass
    ``":memory:"`` for an ephemeral, test-only store.
    """

    def __init__(self, path: str | Path = ".conformguard/calibration.db"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calibration_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                context_bucket TEXT NOT NULL DEFAULT 'default',
                score REAL NOT NULL,
                outcome INTEGER NOT NULL,
                labeling_source TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                calibration_set_version TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_calibration_lookup "
            "ON calibration_examples (tool_name, context_bucket, calibration_set_version)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> CalibrationStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def add(self, example: CalibrationExample) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO calibration_examples
                (tool_name, context_bucket, score, outcome, labeling_source, timestamp, calibration_set_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                example.tool_name,
                example.context_bucket,
                example.score,
                int(example.outcome),
                example.labeling_source.value,
                example.timestamp.astimezone(timezone.utc).isoformat(),
                example.calibration_set_version,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def add_many(self, examples: Sequence[CalibrationExample]) -> list[int]:
        return [self.add(example) for example in examples]

    def query(
        self,
        tool_name: str | None = None,
        context_bucket: str | None = None,
        outcome: bool | None = None,
        calibration_set_version: str | None = None,
    ) -> list[CalibrationExample]:
        clauses: list[str] = []
        params: list[object] = []
        if tool_name is not None:
            clauses.append("tool_name = ?")
            params.append(tool_name)
        if context_bucket is not None:
            clauses.append("context_bucket = ?")
            params.append(context_bucket)
        if outcome is not None:
            clauses.append("outcome = ?")
            params.append(int(outcome))
        if calibration_set_version is not None:
            clauses.append("calibration_set_version = ?")
            params.append(calibration_set_version)

        sql = "SELECT * FROM calibration_examples"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp ASC"

        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_example(row) for row in rows]

    def count(self, **filters: object) -> int:
        return len(self.query(**filters))  # type: ignore[arg-type]

    def timestamp_range(self, **filters: object) -> tuple[datetime, datetime] | None:
        examples = self.query(**filters)  # type: ignore[arg-type]
        if not examples:
            return None
        timestamps = [example.timestamp for example in examples]
        return min(timestamps), max(timestamps)

    def staleness_warning(
        self,
        max_age: timedelta = DEFAULT_STALENESS_THRESHOLD,
        now: datetime | None = None,
        **filters: object,
    ) -> str | None:
        """Return a warning string if the newest example is older than ``max_age``, else None."""
        time_range = self.timestamp_range(**filters)  # type: ignore[arg-type]
        if time_range is None:
            return None
        _, newest = time_range
        reference_now = now or datetime.now(timezone.utc)
        age = reference_now - newest
        if age > max_age:
            return (
                f"Calibration data is stale: newest example is {age.days} days old, "
                f"exceeding the {max_age.days}-day staleness threshold. Exchangeability "
                f"requires the calibration set to represent the current deployment "
                f"distribution -- recalibrate before trusting this guarantee."
            )
        return None

    def size_warning(
        self,
        recommended_minimum: int = RECOMMENDED_MINIMUM_SIZE,
        **filters: object,
    ) -> str | None:
        """Return a warning string if the matching example count is below the recommended minimum."""
        n = self.count(**filters)  # type: ignore[arg-type]
        if n < recommended_minimum:
            return (
                f"Calibration set has {n} examples, below the recommended minimum of "
                f"{recommended_minimum}. Coverage fluctuation widens as calibration size "
                f"shrinks (Angelopoulos & Bates, arXiv:2107.07511, Table 1) -- treat this "
                f"guarantee as provisional until more examples are collected."
            )
        return None


def _row_to_example(row: sqlite3.Row) -> CalibrationExample:
    return CalibrationExample(
        id=row["id"],
        tool_name=row["tool_name"],
        context_bucket=row["context_bucket"],
        score=row["score"],
        outcome=bool(row["outcome"]),
        labeling_source=LabelingSource(row["labeling_source"]),
        timestamp=datetime.fromisoformat(row["timestamp"]),
        calibration_set_version=row["calibration_set_version"],
    )
