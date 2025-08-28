from __future__ import annotations
from supabase import create_client, Client
from .config import get_settings
import sqlite3
from pathlib import Path


def get_supabase() -> Client:
    """Return a configured Supabase client using environment variables."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_key)


def fetch_table(client: Client, table: str):
    """Fetch all rows from a table. Returns list of dicts."""
    res = client.table(table).select('*').execute()
    return res.data or []


def upsert(client: Client, table: str, payload: dict, on_conflict: str | None = None):
    q = client.table(table).upsert(payload)
    if on_conflict:
        q = q.on_conflict(on_conflict)
    return q.execute()


# SQLite helpers
def get_sqlite(db_path: str | Path = "scores.db") -> sqlite3.Connection:
    p = Path(db_path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_sqlite_scores_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scores (
            id TEXT PRIMARY KEY,
            application_id TEXT UNIQUE NOT NULL,
            completeness REAL,
            fit REAL,
            final REAL,
            recommendation TEXT,
            details TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        """
    )
    # Trigger-like updated_at via trigger (optional); fallback handled in upsert
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS scores_application_id_uniq ON scores(application_id);
        """
    )
    conn.commit()


def upsert_sqlite_score(conn: sqlite3.Connection, application_id: str, payload: dict):
    import json
    cur = conn.cursor()
    # generate id if not exists
    cur.execute("SELECT id FROM scores WHERE application_id = ?", (application_id,))
    row = cur.fetchone()
    sid = row["id"] if row else None
    if not sid:
        # simple random id
        import uuid
        sid = str(uuid.uuid4())
    cur.execute(
        """
        INSERT INTO scores (id, application_id, completeness, fit, final, recommendation, details, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM scores WHERE application_id = ?), datetime('now')), datetime('now'))
        ON CONFLICT(application_id) DO UPDATE SET
            completeness=excluded.completeness,
            fit=excluded.fit,
            final=excluded.final,
            recommendation=excluded.recommendation,
            details=excluded.details,
            updated_at=datetime('now');
        """,
        (
            sid,
            application_id,
            float(payload.get("completeness", 0.0)),
            float(payload.get("fit", 0.0)),
            float(payload.get("final", 0.0)),
            payload.get("recommendation", ""),
            json.dumps(payload.get("details", {}), ensure_ascii=False),
            application_id,
        ),
    )
    conn.commit()
    return {"status": "ok", "application_id": application_id, "id": sid}
