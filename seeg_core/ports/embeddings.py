from __future__ import annotations
from typing import Protocol
import numpy as np


class EmbeddingsProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed_text(self, text: str) -> np.ndarray: ...
