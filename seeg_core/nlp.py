from __future__ import annotations
import functools
import hashlib
import numpy as np

try:
    import spacy  # type: ignore
except Exception:  # spaCy may be unavailable on target platform
    spacy = None  # type: ignore[assignment]


@functools.lru_cache(maxsize=1)
def get_nlp():
    """Return a loaded spaCy model if available; else None.

    We try 'fr_core_news_md'. If it is not installed or spaCy is missing,
    return None so callers can use a lightweight fallback.
    """
    if spacy is None:
        return None
    try:
        return spacy.load('fr_core_news_md')
    except Exception:
        # As a light fallback, try a blank French pipeline (no vectors)
        try:
            return spacy.blank('fr')  # type: ignore[attr-defined]
        except Exception:
            return None


def _hashing_vector(text: str, dim: int = 300) -> np.ndarray:
    """Deterministic hashing-based vector for text when spaCy vectors are unavailable.
    Simple bag-of-words with hashing trick into a fixed-size float32 vector.
    """
    v = np.zeros((dim,), dtype=np.float32)
    if not text:
        return v
    for tok in str(text).lower().split():
        h = int(hashlib.md5(tok.encode('utf-8', errors='ignore')).hexdigest(), 16)
        idx = h % dim
        v[idx] += 1.0
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v


def text_vector(text: str) -> np.ndarray:
    nlp = get_nlp()
    s = (text or "").strip()
    if nlp is None:
        return _hashing_vector(s, dim=300)
    if not s:
        size = int(getattr(getattr(nlp, 'vocab', None), 'vectors_length', 300) or 300)
        return np.zeros((size,), dtype=float)
    try:
        doc = nlp(s)
        vec = getattr(doc, 'vector', None)
        if vec is None or (isinstance(vec, np.ndarray) and vec.size == 0):
            return _hashing_vector(s, dim=300)
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec
    except Exception:
        return _hashing_vector(s, dim=300)


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
