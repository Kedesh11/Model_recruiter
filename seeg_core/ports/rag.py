from __future__ import annotations
from typing import Protocol, List, Dict, Any, Optional


class RAGStore(Protocol):
    def upsert(self, ids: List[str], documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None: ...

    def query(self, query_text: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None, where_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...

    def delete_by_app(self, application_id: str) -> None: ...
