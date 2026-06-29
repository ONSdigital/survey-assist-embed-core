"""Embedding and retrieval helpers backed by pluggable vector-store adapters."""

# pylint: disable=too-many-instance-attributes

import logging
from typing import cast

from autocorrect import Speller

from survey_assist_embed_core.adapters.classifai import ClassifaiVectorBackend
from survey_assist_embed_core.adapters.storage import LocalGcsStorage
from survey_assist_embed_core.models import (
    EmbeddingStatus,
    SearchIndexItem,
    SearchIndexResponse,
)
from survey_assist_embed_core.ports import Storage, VectorBackend, VectorIndex

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
        storage: Storage | None = None,
    ):
        """Initialise the handler for a pre-built vector store."""
        self.k_matches = k_matches
        self.db_dir = db_dir
        self._backend = backend if backend is not None else ClassifaiVectorBackend()
        self._storage = storage if storage is not None else LocalGcsStorage()

        self.spell = Speller()

        self.vector_store, self.index_source_file = self._load_existing_vector_store()

        logger.info("Using vector backend config: %s", self._backend.config)

        self.index_size = (
            self.vector_store.num_vectors if self.vector_store.num_vectors else 0
        )

        logger.info(
            "EmbeddingHandler initialised with config: %s", self.get_embed_config()
        )

    def _load_existing_vector_store(self) -> tuple[VectorIndex, str | None]:
        """Load an existing vector store from a local folder or a GCS URI."""
        logger.info("Loading existing vector store from %s", self.db_dir)
        db_dir = self._storage.resolve_store_path(path=self.db_dir)

        try:
            vector_store, index_source_file = self._backend.load(folder_path=db_dir)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc} Build the vector-store artifacts before initialising "
                f"EmbeddingHandler."
            ) from exc

        logger.info("Existing vector store loaded successfully from %s", self.db_dir)
        return vector_store, index_source_file

    def search_index(self, query: str) -> SearchIndexResponse:
        """Return the nearest index entries for a query string."""
        n_results = min(self.index_size, self.k_matches)
        rows = self.vector_store.search(query, limit=n_results)

        return SearchIndexResponse(
            results=[
                SearchIndexItem(
                    distance=float(1.0 - float(cast(float | int, row["score"]))),
                    title=str(row["doc_text"]),
                    code=str(row["doc_label"]),
                )
                for row in rows
            ]
        )

    def search_index_multi(self, query: list[str | None]) -> SearchIndexResponse:
        """Return the nearest index entries for combined query fields."""
        query_terms = [value for value in query if value is not None]
        if not query_terms:
            return SearchIndexResponse(results=[])

        search_terms_list: set[str] = set()
        for i in range(1, len(query_terms) + 1):
            term = " ".join(query_terms[:i])
            search_terms_list.add(term)
            search_terms_list.add(self.spell(term))
        short_list = [
            hit
            for term in search_terms_list
            for hit in self.search_index(query=term).results
        ]
        return SearchIndexResponse(
            results=sorted(short_list, key=lambda item: item.distance)
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
