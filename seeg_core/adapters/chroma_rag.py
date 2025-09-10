from __future__ import annotations
from typing import Dict, Any, List, Optional

from seeg_core.ports.rag import RAGStore
from seeg_core.rag import ChromaRAG


class ChromaRAGAdapter(RAGStore):
    """Adapter RAGStore basé sur ChromaRAG interne."""

    def __init__(self, collection_name: str = "seeg_rag", persist_dir: Optional[str] = None, model_name: Optional[str] = None) -> None:
        # ChromaRAG lit déjà ST_MODEL et CHROMA_DIR via os.environ si None
        self._rag = ChromaRAG(
            collection_name=collection_name,
            persist_dir=persist_dir or ChromaRAG.__init__.__defaults__[1],  # type: ignore[index]
            model_name=model_name or ChromaRAG.__init__.__defaults__[2],  # type: ignore[index]
        )

    def upsert(self, ids: List[str], documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        self._rag.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query_text: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None, where_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._rag.query(query_text=query_text, n_results=n_results, where=where, where_document=where_document)

    def delete_by_app(self, application_id: str) -> None:
        self._rag.delete_by_app(application_id)
