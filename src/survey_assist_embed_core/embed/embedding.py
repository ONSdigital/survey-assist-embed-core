"""Embedding and retrieval helpers backed by pluggable vector-store adapters."""

# pylint: disable=too-many-instance-attributes

import logging
from typing import cast

from autocorrect import Speller

from survey_assist_embed_core.adapters.classifai import ClassifaiVectorBackend
from survey_assist_embed_core.adapters.storage import (
    download_vector_store_from_gcs,
    is_gcs_path,
)
from survey_assist_embed_core.models import (
    EmbeddingStatus,
    SearchIndexItem,
    SearchIndexResponse,
)
from survey_assist_embed_core.ports import SearchRow, VectorBackend, VectorIndex

DEFAULT_DB_DIR = "vector_store"
DEFAULT_K_MATCHES = 20

logger = logging.getLogger(__name__)


class EmbeddingHandler:
    """Handle embedding operations for a vector-store backend."""

    def __init__(
        self,
        db_dir: str = DEFAULT_DB_DIR,
        k_matches: int = DEFAULT_K_MATCHES,
        *,
        backend: VectorBackend | None = None,
    ):
        """Initialise the handler for a pre-built vector store."""
        self.k_matches = k_matches
        self.db_dir = db_dir
        self._backend = backend if backend is not None else ClassifaiVectorBackend()

        self.spell = Speller()

        self.vector_store, self.index_source_file = self._load_existing_vector_store()

        logger.info("Using vector backend config: %s", self._backend.config)

        self.index_size = (
            self.vector_store.num_vectors if self.vector_store.num_vectors else 0
        )
        if self.index_size < 1:
            raise ValueError(
                "Persisted vector store contains no vectors. Rebuild the "
                "vector-store artifacts before initialising EmbeddingHandler."
            )

        logger.info(
            "EmbeddingHandler initialised with config: %s", self.get_embed_config()
        )

    def search_index(self, query: str) -> SearchIndexResponse:
        """Return the nearest index entries for a query string."""
        n_results = min(self.index_size, self.k_matches)
        rows = self.vector_store.search(query, limit=n_results)

        return SearchIndexResponse(results=self._rows_to_search_items(rows))

    def search_index_multi(self, query: list[str | None]) -> SearchIndexResponse:
        """Return the nearest index entries for combined query fields."""
        query_terms = [value for value in query if value is not None]
        if not query_terms:
            return SearchIndexResponse(results=[])

        search_terms: set[str] = set()
        for i in range(1, len(query_terms) + 1):
            term = " ".join(query_terms[:i])
            search_terms.add(term)
            search_terms.add(self.spell(term))
        n_results = min(self.index_size, self.k_matches)
        short_list = [
            item
            for rows in self.vector_store.search_many(
                queries=sorted(search_terms),
                limit=n_results,
            )
            for item in self._rows_to_search_items(rows)
        ]
        return SearchIndexResponse(
            results=self._sort_and_deduplicate_results(short_list)
        )

    def get_embed_config(self) -> EmbeddingStatus:
        """Return the current embedding configuration and ready status."""
        return EmbeddingStatus(
            db_dir=self.db_dir,
            k_matches=self.k_matches,
            index_source_file=self.index_source_file,
            backend=self._backend.config,
            index_size=self.index_size,
            status="ready",
        )

    def _load_existing_vector_store(self) -> tuple[VectorIndex, str | None]:
        """Load an existing vector store from a local folder or a GCS URI."""
        logger.info("Loading existing vector store from %s", self.db_dir)
        if is_gcs_path(self.db_dir):
            with download_vector_store_from_gcs(self.db_dir) as downloaded:
                return self._load_vector_store_from_path(folder_path=downloaded.path)

        return self._load_vector_store_from_path(folder_path=self.db_dir)

    def _load_vector_store_from_path(
        self, *, folder_path: str
    ) -> tuple[VectorIndex, str | None]:
        """Load a vector store from an already-resolved local folder path."""
        try:
            vector_store, index_source_file = self._backend.load(
                folder_path=folder_path
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc} Build the vector-store artifacts before initialising "
                f"EmbeddingHandler."
            ) from exc

        logger.info("Existing vector store loaded successfully from %s", self.db_dir)
        return vector_store, index_source_file

    def _rows_to_search_items(self, rows: list[SearchRow]) -> list[SearchIndexItem]:
        """Convert backend search rows into the public response shape."""
        return [
            SearchIndexItem(
                distance=float(1.0 - float(cast(float | int, row["score"]))),
                title=str(row["doc_text"]),
                code=str(row["doc_label"]),
            )
            for row in rows
        ]

    def _sort_and_deduplicate_results(
        self, items: list[SearchIndexItem]
    ) -> list[SearchIndexItem]:
        """Deduplicate combined-search results and return them in a stable order."""
        best_by_key: dict[tuple[str, str], SearchIndexItem] = {}
        for item in items:
            item_key = (item.code, item.title)
            existing = best_by_key.get(item_key)
            if existing is None or item.distance < existing.distance:
                best_by_key[item_key] = item

        return sorted(best_by_key.values(), key=self._search_item_sort_key)

    def _search_item_sort_key(
        self, item: SearchIndexItem
    ) -> tuple[float, str, str, str]:
        """Return a deterministic sort key for public search items."""
        return (item.distance, item.code, item.title.casefold(), item.title)
