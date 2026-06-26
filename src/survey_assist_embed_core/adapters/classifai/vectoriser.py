"""ClassifAI vectoriser helpers for retrieval backends."""

import numpy as np
from classifai.vectorisers import HuggingFaceVectoriser


class ChromaDBesqueHFVectoriser(HuggingFaceVectoriser):
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

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents into normalised vectors."""
        return self.transform(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query into a normalised vector."""
        return self.transform([text]).tolist()[0]

    def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents into normalised vectors."""
        return self.embed_documents(texts)

    def aembed_query(self, text: str) -> list[float]:
        """Embed a single query into a normalised vector."""
        return self.embed_query(text)
