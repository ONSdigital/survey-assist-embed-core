"""ClassifAI vectoriser helpers for retrieval backends."""

# pylint: disable=too-few-public-methods
from enum import StrEnum
from threading import Lock

import numpy as np
from classifai.vectorisers import HuggingFaceVectoriser, VectoriserBase
from light_embed import TextEmbedding

_ONNX_MODEL_CACHE: dict[tuple[str, str | None], TextEmbedding] = {}
_ONNX_MODEL_CACHE_LOCK = Lock()

type VectoriserClassLike = str | VectoriserClass | None


class VectoriserClass(StrEnum):
    """Canonical selector values for supported semantic vectorisers."""

    ONNX = "onnx"
    HUGGINGFACE = "huggingface"

    @classmethod
    def _missing_(cls, value: object) -> "VectoriserClass | None":
        """Accept common alias spellings for vectoriser selection."""
        if not isinstance(value, str):
            return None
        normalised_class = (
            value.strip()
            .casefold()
            .replace("_", "")
            .replace("-", "")
            .removesuffix("vectoriser")
            .removesuffix("vectorizer")
        )
        if normalised_class == "onnx":
            return cls.ONNX
        if normalised_class in {
            "hf",
            "huggingface",
            "normalisedhf",
            "normalizedhf",
        }:
            return cls.HUGGINGFACE
        return None

    @classmethod
    def resolve(cls, value: VectoriserClassLike) -> "VectoriserClass":
        """Resolve optional loose caller input into a canonical enum value."""
        if value is None:
            return _DEFAULT_VECTORIZER_CLASS
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                "vectoriser_class must resolve to either 'onnx' or 'huggingface'"
            ) from exc


_DEFAULT_VECTORIZER_CLASS = VectoriserClass.ONNX


def _get_cached_onnx_model(model: str, *, device: str | None = None) -> TextEmbedding:
    """Return a process-local cached ONNX embedding model."""
    cache_key = (model, device)
    cached_model = _ONNX_MODEL_CACHE.get(cache_key)
    if cached_model is not None:
        return cached_model

    with _ONNX_MODEL_CACHE_LOCK:
        cached_model = _ONNX_MODEL_CACHE.get(cache_key)
        if cached_model is None:
            cached_model = TextEmbedding(model_name_or_path=model, device=device)
            _ONNX_MODEL_CACHE[cache_key] = cached_model

    return cached_model


class OnnxVectoriser(VectoriserBase):
    """Sentence embedding vectoriser using light_embed with ONNX backend.

    Supports both HuggingFace model names and local paths. The model is cached
    at module level so initialization only happens once per process.
    Outputs are L2-normalised.
    """

    def __init__(self, model: str, *, device: str | None = None) -> None:
        """Load or retrieve cached TextEmbedding model."""
        self.model = _get_cached_onnx_model(model, device=device)
        self.model_name = model
        self.device = device

    def transform(self, texts: list[str] | str) -> np.ndarray:
        """Encode texts and return L2-normalised embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        vectors = np.asarray(list(self.model.encode(texts)), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        return normalise_vectors(vectors)


def normalise_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise a 2D array of vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


class NormalisedHFVectoriser(HuggingFaceVectoriser):
    """HuggingFace vectoriser that normalises embeddings to unit length."""

    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        """Initialise the wrapped Hugging Face model on the requested device."""
        super().__init__(model_name=model_name, device=device)

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
        return normalise_vectors(vectors)


def build_vectoriser(
    embedding_model_name: str,
    *,
    vectoriser_class: VectoriserClassLike = None,
) -> VectoriserBase:
    """Construct a concrete vectoriser for the selected backend kind."""
    if VectoriserClass.resolve(vectoriser_class) == VectoriserClass.ONNX:
        return OnnxVectoriser(model=embedding_model_name)
    return NormalisedHFVectoriser(model_name=embedding_model_name)
