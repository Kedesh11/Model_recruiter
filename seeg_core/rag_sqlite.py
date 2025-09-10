from __future__ import annotations
from typing import List, Dict

import numpy as np

from .config_shim import get_settings
from .embeddings import get_embeddings


def ensure_sqlite_rag_schema(conn) -> None:
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_index (
                application_id TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                text TEXT,
                embedding BLOB,
                PRIMARY KEY (application_id, chunk_id)
            )
            """
        )
        conn.commit()
    except Exception:
        pass


def upsert_rag_for_application(conn, application_id: str, full_text: str, chunk_size: int = 1200, overlap: int = 200) -> None:
    if not application_id or not full_text:
        return
    ensure_sqlite_rag_schema(conn)

    # découpage naïf par caractères, cohérent avec app
    def _chunk_text(t: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
        if not t:
            return []
        s = " ".join(str(t).split())
        if len(s) <= max_chars:
            return [s]
        chunks: List[str] = []
        start = 0
        while start < len(s):
            end = min(len(s), start + max_chars)
            chunks.append(s[start:end])
            if end >= len(s):
                break
            start = max(0, end - overlap)
        return chunks

    chunks = _chunk_text(full_text, max_chars=chunk_size, overlap=overlap)

    settings = get_settings()
    api_key = getattr(settings, "openai_api_key", "")
    embs = get_embeddings(chunks, api_key=api_key, model="text-embedding-3-small", dim=256)

    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM rag_index WHERE application_id = ?", (str(application_id),))
        for i, (txt, vec) in enumerate(zip(chunks, embs)):
            arr = np.array(vec, dtype=np.float32)
            cur.execute(
                "INSERT OR REPLACE INTO rag_index(application_id, chunk_id, text, embedding) VALUES (?, ?, ?, ?)",
                (str(application_id), int(i), txt, arr.tobytes()),
            )
        conn.commit()
    except Exception:
        pass


def rag_search(conn, application_id: str, query: str, top_k: int = 3) -> List[Dict]:
    if not application_id or not query:
        return []
    ensure_sqlite_rag_schema(conn)

    settings = get_settings()
    api_key = getattr(settings, "openai_api_key", "")
    q_embs = get_embeddings([query], api_key=api_key, model="text-embedding-3-small", dim=256)
    if not q_embs:
        return []
    q = np.array(q_embs[0], dtype=np.float32)

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    try:
        cur = conn.cursor()
        cur.execute("SELECT chunk_id, text, embedding FROM rag_index WHERE application_id = ?", (str(application_id),))
        rows = cur.fetchall()
        scored: List[tuple[float, int, str]] = []
        for cid, txt, blob in rows:
            try:
                v = np.frombuffer(blob, dtype=np.float32)
                scored.append((float(_cosine(q, v)), int(cid), txt))
            except Exception:
                continue
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict] = []
        for s, cid, txt in scored[:max(1, int(top_k))]:
            out.append({"chunk_id": cid, "score": s, "text": txt})
        return out
    except Exception:
        return []
