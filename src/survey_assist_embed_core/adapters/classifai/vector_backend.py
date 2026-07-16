"""ClassifAI implementation of the vector backend port."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

from classifai.indexers import (
    VectorStore,
    VectorStoreSearchInput,
    VectorStoreSearchOutput,
)
from survey_assist_utils.logging import get_logger

from survey_assist_embed_core.adapters.classifai.artifacts import (
    ensure_persisted_vector_store,
    read_embedding_model_name,
    read_index_source_file,
    write_vector_store_metadata,
)
from survey_assist_embed_core.adapters.classifai.vectoriser import (
    NormalisedHFVectoriser,
    OnnxVectoriser,
    VectoriserBase,
)
from survey_assist_embed_core.adapters.storage import (
    download_one_file_from_gcs,
    is_gcs_path,
)
from survey_assist_embed_core.models import VectorBackendConfig
from survey_assist_embed_core.ports import SearchRow, VectorIndex

logger = get_logger(__name__)

_DEFAULT_SENTENCE_TRANSFORMERS_ORG = "sentence-transformers"
DEFAULT_CLASSIFAI_EMBEDDING_MODEL_NAME = (
    f"{_DEFAULT_SENTENCE_TRANSFORMERS_ORG}/all-MiniLM-L6-v2"
)

_DEFAULT_VECTORIZER_CLASS = "ONNX"  # alternatively "HF" for Hugging Face vectoriser


def build_classifai_vector_store_artifacts(
    *,
    index_source_file: str,
    output_dir: str,
    embedding_model_name: str | None = None,
    vectoriser_class: str | None = None,
) -> None:
    """Build persisted ClassifAI vector-store artifacts from a source file.

    Args:
        index_source_file: Local path or GCS URI for the CSV source data.
        output_dir: Directory where the persisted vector-store artifacts are
            written.
        embedding_model_name: Embedding model name or fully qualified
            HuggingFace identifier to use during vectorisation.
        vectoriser_class: Optional vectoriser kind to use for the persisted
            vector store.  If not provided, the default ONNX vectoriser is used.
    """
    embedding_model_name = resolve_model_name(embedding_model_name)
    vectoriser_class = resolve_vectoriser_class(vectoriser_class)
    logger.info(
        "Starting vector store artifact build",
        embedding_model_name=embedding_model_name,
        output_dir=output_dir,
    )
    with _resolve_local_path(index_source_file) as local_file:
        vectoriser = _build_vectoriser(
            embedding_model_name, vectoriser_class=vectoriser_class
        )
        VectorStore(
            file_name=local_file,
            data_type="csv",
            vectoriser=vectoriser,
            batch_size=8,
            meta_data=None,
            output_dir=output_dir,
            overwrite=True,
            hooks=None,
        )

    write_vector_store_metadata(
        folder_path=output_dir,
        index_source_file=index_source_file,
        embedding_model_name=embedding_model_name,
    )
    logger.info(
        "Vector store artifacts built successfully",
        output_dir=output_dir,
        embedding_model_name=embedding_model_name,
    )


class _ClassifaiVectorIndex:
    """Adapter around a ClassifAI vector store instance."""

    def __init__(self, store: VectorStore):
        """Store the wrapped ClassifAI vector store."""
        self._store = store

    @property
    def num_vectors(self) -> int | None:
        """Return the number of vectors in the wrapped store."""
        return self._store.num_vectors

    def search(self, query: str, *, limit: int) -> list[SearchRow]:
        """Search the wrapped store and return backend rows."""
        return self.search_many([query], limit=limit)[0]

    def search_many(self, queries: list[str], *, limit: int) -> list[list[SearchRow]]:
        """Search the wrapped store for multiple queries in one backend call."""
        if not queries:
            return []

        query_ids = [f"q{i}" for i in range(1, len(queries) + 1)]
        search_input = VectorStoreSearchInput({"id": query_ids, "query": queries})
        results: VectorStoreSearchOutput = self._store.search(
            search_input,
            n_results=limit,
        )
        rows_by_query_id: dict[str, list[SearchRow]] = {
            query_id: [] for query_id in query_ids
        }
        for row in cast(list[SearchRow], results.to_dict(orient="records")):
            rows_by_query_id[str(row["query_id"])].append(row)

        return [rows_by_query_id[query_id] for query_id in query_ids]


class ClassifaiVectorBackend:
    """Vector backend that loads persisted ClassifAI search artifacts.

    The backend reads persisted metadata, constructs the query vectoriser, and
    returns a loaded index that satisfies the vector-backend protocol.
    """

    def __init__(self):
        """Initialise an unloaded backend waiting for persisted metadata."""
        self._embedding_model_name: str | None = None
        self._vectoriser_class: str | None = None
        self._vectoriser: VectoriserBase | None = None

    @property
    def config(self) -> VectorBackendConfig:
        """Return typed backend metadata for handler status output."""
        return VectorBackendConfig(
            backend_name="classifai",
            settings={
                "embedding_model_name": self._embedding_model_name,
                "vectoriser_class": self._vectoriser_class,
            },
        )

    def load(
        self,
        *,
        folder_path: str,
        vectoriser_class: str | None = None,
    ) -> tuple[VectorIndex, str | None]:
        """Load a persisted ClassifAI vector store from a local folder.

        Args:
            folder_path: Local folder containing the persisted vector-store
                artifacts.
            vectoriser_class: Optional vectoriser kind to use for the loaded
                vector store.  If not provided, the default ONNX vectoriser is used.

        Returns:
            A tuple of the loaded vector index and the recorded source-file
            path, if available.
        """
        ensure_persisted_vector_store(folder_path=folder_path)
        embedding_model_name = read_embedding_model_name(folder_path=folder_path)
        self._set_embedding_model_name(embedding_model_name)
        if embedding_model_name is None:
            logger.warning(
                "No embedding model metadata found in persisted vector store. Using default model.",
                folder_path=folder_path,
                resolved_model_name=self._embedding_model_name,
            )

        self._set_vectoriser_class(vectoriser_class)
        vectoriser = self._get_vectoriser()
        store = VectorStore.from_filespace(
            folder_path=folder_path,
            vectoriser=vectoriser,
            hooks=None,
        )
        index_source_file = read_index_source_file(folder_path=folder_path)
        return _ClassifaiVectorIndex(store), index_source_file

    def _set_embedding_model_name(self, embedding_model_name: str | None) -> None:
        """Update the effective embedding model and clear any stale cache."""
        resolved_name = resolve_model_name(embedding_model_name)
        if self._embedding_model_name == resolved_name:
            return

        self._embedding_model_name = resolved_name
        self._vectoriser = None

    def _set_vectoriser_class(self, vectoriser_class: str | None) -> None:
        """Update the effective vectoriser class and clear any stale cache."""
        resolved_class = resolve_vectoriser_class(vectoriser_class)
        if self._vectoriser_class == resolved_class:
            return

        self._vectoriser_class = resolved_class
        self._vectoriser = None

    def _get_vectoriser(self) -> VectoriserBase:
        """Build and cache the default ClassifAI vectoriser."""
        if self._vectoriser is not None:
            return self._vectoriser

        self._vectoriser = _build_vectoriser(
            embedding_model_name=self._embedding_model_name,
            vectoriser_class=self._vectoriser_class,
        )

        return self._vectoriser


def _build_vectoriser(
    embedding_model_name: str | None = None,
    *,
    vectoriser_class: str | None = None,
) -> VectoriserBase:
    """Construct a concrete vectoriser for the selected backend kind."""
    if resolve_vectoriser_class(vectoriser_class) == "ONNX":
        return OnnxVectoriser(model=resolve_model_name(embedding_model_name))
    return NormalisedHFVectoriser(model_name=resolve_model_name(embedding_model_name))


def resolve_model_name(name: str | None) -> str:
    """Return a fully-qualified HuggingFace model identifier.

    If ``name`` already contains a ``/`` it is treated as a complete
    ``{org}/{model}`` identifier and returned unchanged.  Otherwise the
    sentence-transformers organisation is prepended as the default.
    """
    if name is None:
        return DEFAULT_CLASSIFAI_EMBEDDING_MODEL_NAME
    if "/" in name:
        return name
    return f"{_DEFAULT_SENTENCE_TRANSFORMERS_ORG}/{name}"


def resolve_vectoriser_class(vectoriser_class: str | None) -> str:
    """Resolve requested vectoriser class into a concrete implementation name."""
    if vectoriser_class is None:
        return _DEFAULT_VECTORIZER_CLASS
    selected_class = str(vectoriser_class).strip()
    normalised_class = (
        selected_class.casefold()
        .replace("_", "")
        .replace("-", "")
        .removesuffix("vectoriser")
        .removesuffix("vectorizer")
    )

    if normalised_class in {"onnx"}:
        return "ONNX"

    if normalised_class in {
        "hf",
        "huggingface",
        "normalisedhf",
        "normalizedhf",
    }:
        return "HF"

    raise ValueError("vectoriser_class must resolve to either 'ONNX' or 'HF'")


@contextmanager
def _resolve_local_path(path: str) -> Iterator[str]:
    """Yield a local filesystem path, downloading from GCS if necessary."""
    if is_gcs_path(path):
        with download_one_file_from_gcs(path) as downloaded:
            yield downloaded.path
    else:
        yield path
