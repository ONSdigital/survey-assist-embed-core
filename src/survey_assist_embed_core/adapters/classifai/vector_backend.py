"""ClassifAI implementation of the vector backend port."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

from classifai.indexers import (
    VectorStore,
    VectorStoreSearchInput,
    VectorStoreSearchOutput,
)

from survey_assist_embed_core.adapters.classifai.artifacts import (
    ensure_persisted_vector_store,
    read_embedding_model_name,
    read_index_source_file,
    write_vector_store_metadata,
)
from survey_assist_embed_core.adapters.classifai.vectoriser import (
    NormalisedHFVectoriser,
)
from survey_assist_embed_core.adapters.storage import (
    download_one_file_from_gcs,
    is_gcs_path,
)
from survey_assist_embed_core.models import VectorBackendConfig
from survey_assist_embed_core.ports import SearchRow, VectorIndex

DEFAULT_CLASSIFAI_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@contextmanager
def _resolve_local_path(path: str) -> Iterator[str]:
    """Yield a local filesystem path, downloading from GCS if necessary."""
    if is_gcs_path(path):
        with download_one_file_from_gcs(path) as downloaded:
            yield downloaded.path
    else:
        yield path


def build_classifai_vector_store_artifacts(
    *,
    index_source_file: str,
    output_dir: str,
    embedding_model_name: str = DEFAULT_CLASSIFAI_EMBEDDING_MODEL_NAME,
) -> None:
    """Build persisted ClassifAI vector-store artifacts from a source CSV."""
    with _resolve_local_path(index_source_file) as local_file:
        vectoriser = NormalisedHFVectoriser(model_name=embedding_model_name)
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
        search_input = VectorStoreSearchInput({"id": ["q1"], "query": [query]})
        results: VectorStoreSearchOutput = self._store.search(
            search_input,
            n_results=limit,
        )
        return cast(list[SearchRow], results.to_dict(orient="records"))


class ClassifaiVectorBackend:
    """ClassifAI implementation of the vector backend port."""

    def __init__(self):
        """Initialise an unloaded backend waiting for persisted metadata."""
        self._embedding_model_name: str | None = None
        self._vectoriser: NormalisedHFVectoriser | None = None

    @property
    def config(self) -> VectorBackendConfig:
        """Return typed backend metadata for handler status output."""
        return VectorBackendConfig(
            backend_name="classifai",
            settings={
                "embedding_model_name": self._embedding_model_name,
            },
        )

    def load(self, *, folder_path: str) -> tuple[VectorIndex, str | None]:
        """Load a ClassifAI vector store from filespace."""
        ensure_persisted_vector_store(folder_path=folder_path)
        embedding_model_name = read_embedding_model_name(folder_path=folder_path)
        if embedding_model_name is None:
            raise FileNotFoundError(
                "No embedding model metadata found in persisted vector store."
            )

        self._set_embedding_model_name(embedding_model_name)

        vectoriser = self._get_vectoriser()
        store = VectorStore.from_filespace(
            folder_path=folder_path,
            vectoriser=vectoriser,
            hooks=None,
        )
        index_source_file = read_index_source_file(folder_path=folder_path)
        return _ClassifaiVectorIndex(store), index_source_file

    def _set_embedding_model_name(self, embedding_model_name: str) -> None:
        """Update the effective embedding model and clear any stale cache."""
        if self._embedding_model_name == embedding_model_name:
            return

        self._embedding_model_name = embedding_model_name
        self._vectoriser = None

    def _get_vectoriser(self) -> NormalisedHFVectoriser:
        """Build and cache the default ClassifAI vectoriser."""
        vectoriser = self._vectoriser
        if vectoriser is None:
            if self._embedding_model_name is None:
                raise ValueError(
                    "embedding_model_name must be loaded from persisted metadata "
                    "before constructing a query vectoriser."
                )
            vectoriser = NormalisedHFVectoriser(model_name=self._embedding_model_name)
            self._vectoriser = vectoriser
        return vectoriser
