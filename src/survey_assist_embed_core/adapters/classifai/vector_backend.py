"""ClassifAI implementation of the vector backend port."""

from typing import Any, cast

from classifai.indexers import (
    VectorStore,
    VectorStoreSearchInput,
    VectorStoreSearchOutput,
)

from survey_assist_embed_core.ports import SearchRow, VectorIndex


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

    def load(self, *, folder_path: str, vectoriser: Any) -> VectorIndex:
        """Load a ClassifAI vector store from filespace."""
        store = VectorStore.from_filespace(
            folder_path=folder_path,
            vectoriser=vectoriser,
            hooks=None,
        )
        return _ClassifaiVectorIndex(store)

    def build(self, *, file_name: str, vectoriser: Any, output_dir: str) -> VectorIndex:
        """Build a ClassifAI vector store from a CSV source file."""
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
