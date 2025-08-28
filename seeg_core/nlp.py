from __future__ import annotations
import functools
import numpy as np
import spacy


@functools.lru_cache(maxsize=1)
def get_nlp():
    return spacy.load('fr_core_news_md')


def text_vector(text: str) -> np.ndarray:
    nlp = get_nlp()
    if not text or str(text).strip() == "":
        # fallback vector size
        size = getattr(nlp.vocab, 'vectors_length', 300)
        return np.zeros((size,), dtype=float)
    doc = nlp(text)
    vec = doc.vector
    # normalize
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        m = min(a.shape[0], b.shape[0])
        a = a[:m]
        b = b[:m]
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
