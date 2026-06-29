"""ClassifAI implementation of the vector backend port."""

from typing import cast

from classifai.indexers import (
    VectorStore,
    VectorStoreSearchInput,
    VectorStoreSearchOutput,
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

    def load(self, *, folder_path: str) -> VectorIndex:
        """Load a ClassifAI vector store from filespace."""
        vectoriser = self._get_vectoriser()
        store = VectorStore.from_filespace(
            folder_path=folder_path,
            vectoriser=vectoriser,
            hooks=None,
        )
        return _ClassifaiVectorIndex(store)

    def build(self, *, file_name: str, output_dir: str) -> VectorIndex:
        """Build a ClassifAI vector store from a CSV source file."""
        vectoriser = self._get_vectoriser()
        store = VectorStore(
            file_name=file_name,
            data_type="csv",
            vectoriser=vectoriser,
            batch_size=8,
            meta_data=None,
            output_dir=output_dir,
            overwrite=True,
            hooks=None,
        )
        return _ClassifaiVectorIndex(store)
