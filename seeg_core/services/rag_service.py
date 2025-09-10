from __future__ import annotations
from functools import lru_cache
from typing import Any, Dict, List, Optional

from seeg_core.adapters.chroma_rag import ChromaRAGAdapter
from seeg_core.ports.rag import RAGStore


class RagService:
    """Service d'accès RAG.

    Fournit une API simple pour indexer et rechercher par application en s'appuyant sur
    un RAGStore (Chroma).
    """

    def __init__(self, store: RAGStore) -> None:
        self._store = store

    def upsert_for_application(self, application_id: str, full_text: str) -> None:
        if not application_id or not full_text:
            return
        chunks = _simple_chunk(full_text, max_chars=1200, overlap=200)
        ids = [f"app:{application_id}:ch:{i}" for i in range(len(chunks))]
        metadatas = [{"application_id": application_id, "chunk_id": i} for i in range(len(chunks))]
        self._store.upsert(ids=ids, documents=chunks, metadatas=metadatas)

    def search(self, application_id: str, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not application_id or not query_text:
            return []
        res = self._store.query(query_text=query_text, n_results=max(1, int(top_k)), where={"application_id": application_id})
        out: List[Dict[str, Any]] = []
        # Chroma retourne des listes imbriquées, normaliser en liste simple
        docs = (res.get("documents") or [[]])[0]
        scores = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        for d, s, m in zip(docs, scores, metas):
            out.append({"text": d, "score": float(1.0 - s) if isinstance(s, (int, float)) else 0.0, "meta": m or {}})
        return out


def _simple_chunk(t: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    s = (t or "").strip()
    if not s:
        return []
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


@lru_cache(maxsize=1)
def get_rag_service() -> Optional[RagService]:
    try:
        store = ChromaRAGAdapter()
        return RagService(store)
    except Exception:
        # chroma non dispo ou erreur d'init
        return None
