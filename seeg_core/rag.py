from __future__ import annotations
import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions

# sentence-transformers model can be customized via env
_DEFAULT_ST_MODEL = os.getenv("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# persistent directory for Chroma collections
_DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", "/app/chroma")


class ChromaRAG:
    """
    Wrapper minimaliste pour ChromaDB visant:
    - initialisation du client persistant
    - création/chargement de collection
    - upsert de documents
    - requêtes de similarité (top-k)
    """

    def __init__(
        self,
        collection_name: str = "seeg_rag",
        persist_dir: str = _DEFAULT_CHROMA_DIR,
        model_name: str = _DEFAULT_ST_MODEL,
    ) -> None:
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        return self._collection

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Upsert des documents (texte brut) avec IDs stables.
        - ids: identifiants uniques (ex: f"app:{application_id}:doc:{k}")
        - documents: contenus textuels
        - metadatas: dictionnaires additionnels (ex: application_id, doc_type)
        """
        if not ids:
            return
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Recherche sémantique top-k sur la collection.
        Retourne le dictionnaire Chroma standard avec documents/distances/metadatas/ids.
        """
        return self._collection.query(
            query_texts=[query_text],
            n_results=max(1, n_results),
            where=where,
            where_document=where_document,
        )

    def delete_by_app(self, application_id: str) -> None:
        """Supprime tous les éléments associés à une application donnée (metadata match)."""
        self._collection.delete(where={"application_id": application_id})
