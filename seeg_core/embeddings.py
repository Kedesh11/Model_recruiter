from __future__ import annotations
from typing import List

import numpy as np

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore


def embed_texts_openai(texts: List[str], api_key: str, model: str = "text-embedding-3-small") -> List[List[float]]:
    """Embeddings via OpenAI. Retourne [] si indisponible/erreur.
    - texts: liste de chaînes
    - api_key: clé OpenAI
    - model: nom du modèle d'embedding
    """
    if not texts:
        return []
    if OpenAI is None or not api_key:
        return []
    try:
        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]
    except Exception:
        return []


def embed_texts_fallback(texts: List[str], dim: int = 256) -> List[List[float]]:
    """Fallback déterministe type hashing/count-vector simple avec normalisation L2.
    - dim: dimension du vecteur de sortie
    """
    vecs: List[List[float]] = []
    for t in texts or []:
        v = np.zeros(dim, dtype=np.float32)
        if t:
            for tok in str(t).lower().split():
                idx = (hash(tok) % dim)
                v[idx] += 1.0
        n = np.linalg.norm(v)
        vecs.append((v / n).tolist() if n > 0 else v.tolist())
    return vecs


def get_embeddings(texts: List[str], api_key: str | None, model: str = "text-embedding-3-small", dim: int = 256) -> List[List[float]]:
    """Essaie OpenAI puis fallback local.
    - api_key peut être None/""; si absent → fallback direct.
    """
    if not texts:
        return []
    embs = embed_texts_openai(texts, api_key or "", model=model)
    if embs:
        return embs
    return embed_texts_fallback(texts, dim=dim)
