"""ClassifAI vectoriser helpers for retrieval backends."""

import numpy as np
from classifai.vectorisers import HuggingFaceVectoriser


# pylint: disable-next=too-few-public-methods
class NormalisedHFVectoriser(HuggingFaceVectoriser):
    """HuggingFace vectoriser that normalises embeddings to unit length."""

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalise row vectors to unit length."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def transform(self, texts: list[str] | str) -> np.ndarray:
        """Transform text into unit-normalised embeddings.

        Args:
            texts: One text string or a list of text strings to embed.

        Returns:
            A NumPy array containing one normalised embedding per input text.
        """
        if isinstance(texts, str):
            texts = [texts]

        vectors = super().transform(texts)
        return self._normalize(vectors)
