"""ClassifAI implementation of the vector backend port."""

from typing import cast

from classifai.indexers import (
    VectorStore,
    VectorStoreSearchInput,
    VectorStoreSearchOutput,
)

from survey_assist_embed_core.adapters.classifai.artifacts import (
    ensure_persisted_vector_store,
    has_persisted_vectors_file,
    read_index_source_file,
    write_index_source_file,
)
from survey_assist_embed_core.adapters.classifai.vectoriser import (
    NormalisedHFVectoriser,
)
from survey_assist_embed_core.models import VectorBackendConfig
from survey_assist_embed_core.ports import SearchRow, VectorIndex

DEFAULT_CLASSIFAI_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


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

    def __init__(
        self,
        *,
        embedding_model_name: str = DEFAULT_CLASSIFAI_EMBEDDING_MODEL_NAME,
    ):
        """Store the embedding model used to construct ClassifAI vectorisers."""
        self._embedding_model_name = embedding_model_name
        self._vectoriser: NormalisedHFVectoriser | None = None

    @property
    def config(self) -> VectorBackendConfig:
        """Return typed backend metadata for handler status output."""
        return VectorBackendConfig(
            backend_name="classifai",
            settings={"embedding_model_name": self._embedding_model_name},
        )

    def _get_vectoriser(self) -> NormalisedHFVectoriser:
        """Build and cache the default ClassifAI vectoriser."""
        vectoriser = self._vectoriser
        if vectoriser is None:
            vectoriser = NormalisedHFVectoriser(
                model_name=f"sentence-transformers/{self._embedding_model_name}"
            )
            self._vectoriser = vectoriser
        return vectoriser

    def has_persisted_store(self, *, folder_path: str) -> bool:
        """Return whether persisted ClassifAI vector-store files already exist."""
        return has_persisted_vectors_file(folder_path=folder_path)

    def load(self, *, folder_path: str) -> tuple[VectorIndex, str | None]:
        """Load a ClassifAI vector store from filespace."""
        ensure_persisted_vector_store(folder_path=folder_path)
        vectoriser = self._get_vectoriser()
        store = VectorStore.from_filespace(
            folder_path=folder_path,
            vectoriser=vectoriser,
            hooks=None,
        )
        index_source_file = read_index_source_file(folder_path=folder_path)
        return _ClassifaiVectorIndex(store), index_source_file

    def build(
        self,
        *,
        file_name: str,
        output_dir: str,
        index_source_file: str | None,
    ) -> None:
        """Build persisted ClassifAI vector-store artifacts from a CSV file."""
        vectoriser = self._get_vectoriser()
        VectorStore(
            file_name=file_name,
            data_type="csv",
            vectoriser=vectoriser,
            batch_size=8,
            meta_data=None,
            output_dir=output_dir,
            overwrite=True,
            hooks=None,
        )
        write_index_source_file(
            folder_path=output_dir,
            index_source_file=index_source_file,
        )
