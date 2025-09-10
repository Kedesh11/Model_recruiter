from __future__ import annotations
from typing import Optional
import sqlite3

from seeg_core.ports.repositories import WriteRepository
from seeg_core.domain.score import Score
from seeg_core.db import get_sqlite, ensure_sqlite_scores_schema, upsert_sqlite_score


class SQLiteWriteRepository(WriteRepository):
    """Implémentation WriteRepository pour persister les Scores dans SQLite."""

    def __init__(self, db_path: str = "scores.db") -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = get_sqlite(self._db_path)
            ensure_sqlite_scores_schema(self._conn)
        return self._conn

    def upsert_score(self, score: Score) -> None:
        payload = {
            "completeness": float(score.completeness),
            "fit": float(score.fit),
            "final": float(score.final),
            "recommendation": score.recommendation or "",
            "details": {},
        }
        upsert_sqlite_score(self.conn, score.application_id, payload)
