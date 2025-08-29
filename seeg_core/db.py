from __future__ import annotations
from supabase import create_client, Client
from .config import get_settings
import sqlite3
from pathlib import Path
import re
import unicodedata


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


# -----------------------------
# MTP (Métier/Talent/Paradigme)
# -----------------------------
def ensure_sqlite_mtp_schema(conn: sqlite3.Connection):
    """Create tables for MTP questions and dimension metadata if they don't exist."""
    cur = conn.cursor()
    # Questions table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS mtp_questions (
            id TEXT PRIMARY KEY,
            code TEXT UNIQUE NOT NULL,
            poste TEXT NOT NULL,
            dimension TEXT NOT NULL CHECK (dimension IN ('metier','talent','paradigme')),
            question_order INTEGER NOT NULL,
            question TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS mtp_questions_code_uniq ON mtp_questions(code);
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS mtp_questions_poste_dim ON mtp_questions(poste, dimension);
        """
    )
    # Dimension metadata (indicators per poste/dimension)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS mtp_dimension_meta (
            id TEXT PRIMARY KEY,
            poste TEXT NOT NULL,
            dimension TEXT NOT NULL CHECK (dimension IN ('metier','talent','paradigme')),
            indicators TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(poste, dimension)
        );
        """
    )
    conn.commit()


def _slugify(text: str) -> str:
    """Lightweight slugify without external deps (ascii, lower, dash)."""
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip('-')
    text = re.sub(r"-+", "-", text)
    return text


def list_mtp_questions(conn: sqlite3.Connection, active_only: bool = False) -> list[dict]:
    cur = conn.cursor()
    if active_only:
        cur.execute(
            "SELECT * FROM mtp_questions WHERE active = 1 ORDER BY poste, dimension, question_order"
        )
    else:
        cur.execute(
            "SELECT * FROM mtp_questions ORDER BY poste, dimension, question_order"
        )
    rows = cur.fetchall()
    return [dict(r) for r in rows]


def upsert_mtp_question(conn: sqlite3.Connection, payload: dict) -> dict:
    """Insert or update a question by unique code.

    Required payload keys: code, poste, dimension, question_order, question
    Optional: active (defaults to 1)
    """
    import uuid
    cur = conn.cursor()
    # find existing id
    cur.execute("SELECT id FROM mtp_questions WHERE code = ?", (payload["code"],))
    row = cur.fetchone()
    qid = row["id"] if row else str(uuid.uuid4())
    cur.execute(
        """
        INSERT INTO mtp_questions (id, code, poste, dimension, question_order, question, active, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, 1), COALESCE((SELECT created_at FROM mtp_questions WHERE code = ?), datetime('now')), datetime('now'))
        ON CONFLICT(code) DO UPDATE SET
            poste=excluded.poste,
            dimension=excluded.dimension,
            question_order=excluded.question_order,
            question=excluded.question,
            active=excluded.active,
            updated_at=datetime('now');
        """,
        (
            qid,
            payload["code"],
            payload["poste"],
            payload["dimension"],
            int(payload["question_order"]),
            payload["question"],
            int(payload.get("active", 1)),
            payload["code"],
        ),
    )
    conn.commit()
    return {"status": "ok", "code": payload["code"], "id": qid}


def delete_mtp_question(conn: sqlite3.Connection, code: str) -> dict:
    cur = conn.cursor()
    cur.execute("DELETE FROM mtp_questions WHERE code = ?", (code,))
    conn.commit()
    return {"status": "ok", "code": code}


def upsert_mtp_dimension_meta(conn: sqlite3.Connection, poste: str, dimension: str, indicators: str | None):
    import uuid
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM mtp_dimension_meta WHERE poste = ? AND dimension = ?",
        (poste, dimension),
    )
    row = cur.fetchone()
    mid = row["id"] if row else str(uuid.uuid4())
    cur.execute(
        """
        INSERT INTO mtp_dimension_meta (id, poste, dimension, indicators, created_at, updated_at)
        VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM mtp_dimension_meta WHERE poste = ? AND dimension = ?), datetime('now')), datetime('now'))
        ON CONFLICT(poste, dimension) DO UPDATE SET
            indicators=excluded.indicators,
            updated_at=datetime('now');
        """,
        (mid, poste, dimension, indicators, poste, dimension),
    )
    conn.commit()
    return {"status": "ok", "id": mid, "poste": poste, "dimension": dimension}


def list_mtp_dimension_meta(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM mtp_dimension_meta ORDER BY poste, dimension")
    rows = cur.fetchall()
    return [dict(r) for r in rows]


def import_mtp_questions_from_json(conn: sqlite3.Connection, data: dict, active: int = 1) -> dict:
    """Ingest questions from the provided JSON structure.

    Expected structure:
    { "postes": [ { "poste": str, "metiers": {...}, "talent": {...}, "paradigme": {...} } ] }
    Each dimension dict contains questionN keys and optionally 'indicateurs'.
    """
    ensure_sqlite_mtp_schema(conn)
    total = 0
    for item in data.get("postes", []):
        poste = item.get("poste", "").strip()
        if not poste:
            continue
        poste_slug = _slugify(poste)
        for dim_key in ("metiers", "talent", "paradigme"):
            dim_val = item.get(dim_key) or {}
            # save indicators if present
            indicators = dim_val.get("indicateurs") if isinstance(dim_val, dict) else None
            dim_name = "metier" if dim_key == "metiers" else dim_key
            if indicators:
                upsert_mtp_dimension_meta(conn, poste, dim_name, indicators)
            # iterate questions in deterministic order
            if isinstance(dim_val, dict):
                # filter keys like question1, question2...
                q_items = [(k, v) for k, v in dim_val.items() if k.lower().startswith("question")]
                # sort by the numeric suffix if present
                def q_order(k: str) -> int:
                    m = re.search(r"(\d+)$", k)
                    return int(m.group(1)) if m else 0

                q_items.sort(key=lambda kv: q_order(kv[0]))
                for idx, (_, qtext) in enumerate(q_items, start=1):
                    if not qtext:
                        continue
                    code = f"{poste_slug}.{dim_name}.q{idx}"
                    payload = {
                        "code": code,
                        "poste": poste,
                        "dimension": dim_name,
                        "question_order": idx,
                        "question": str(qtext).strip(),
                        "active": int(active),
                    }
                    upsert_mtp_question(conn, payload)
                    total += 1
    return {"status": "ok", "imported": total}
