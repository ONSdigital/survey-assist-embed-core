"""ClassifAI vectoriser helpers for retrieval backends."""

import numpy as np
from classifai.vectorisers import HuggingFaceVectoriser


class NormalisedHFVectoriser(HuggingFaceVectoriser):
    """Normalise HuggingFace vectors to unit length after embedding."""

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalise row vectors to unit length."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def transform(self, texts: list[str] | str) -> np.ndarray:
        """Transform texts into normalised vectors."""
        if isinstance(texts, str):
            texts = [texts]

        vectors = super().transform(texts)
        return self._normalize(vectors)
