"""Embedding and retrieval helpers backed by a ClassifAI vector store."""

# pylint: disable=too-many-instance-attributes

import logging
from typing import cast

import numpy as np
from autocorrect import Speller
from classifai.vectorisers import HuggingFaceVectoriser

# Next PR: Refactor to remove dependency on ClassifAI and GCS and allow
# for pluggable vector store and storage backends.
from survey_assist_embed_core.adapters.classifai import (
    ClassifaiVectorBackend,
    has_persisted_vector_store,
    has_persisted_vectors_file,
    read_index_source_file,
    write_index_source_file,
)
from survey_assist_embed_core.adapters.storage import (
    DownloadedVectorStore,
    download_one_file_from_gcs,
    download_vector_store_from_gcs,
    is_gcs_path,
)
from survey_assist_embed_core.models import (
    EmbeddingStatus,
    SearchIndexItem,
    SearchIndexResponse,
)
from survey_assist_embed_core.ports import VectorBackend, VectorIndex

DEFAULT_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_DB_DIR = "vector_store"
DEFAULT_K_MATCHES = 20

logger = logging.getLogger(__name__)


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


class EmbeddingHandler:
    """Handle embedding operations for a vector-store backend."""

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        db_dir: str = DEFAULT_DB_DIR,
        k_matches: int = DEFAULT_K_MATCHES,
        index_source_file: str | None = None,
        backend: VectorBackend | None = None,
    ):
        """Initialise the handler for an existing or newly built vector store."""
        self.embedding_model_name = embedding_model_name
        self.k_matches = k_matches
        self.db_dir = db_dir
        self.index_source_file = index_source_file
        self._backend = backend or ClassifaiVectorBackend()

        self.embeddings: HuggingFaceVectoriser = ChromaDBesqueHFVectoriser(
            model_name=f"sentence-transformers/{embedding_model_name}"
        )
        logger.info("Using embedding model: %s", embedding_model_name)

        self.spell = Speller()

        self._downloaded_vector_store: DownloadedVectorStore | None = None
        self.vector_store: VectorIndex
        if not self.index_source_file:
            self.vector_store, self.index_source_file = (
                self._load_existing_vector_store()
            )
        else:
            self.vector_store = self._build_vector_store()

        self.index_size = (
            self.vector_store.num_vectors if self.vector_store.num_vectors else 0
        )

        logger.info(
            "EmbeddingHandler initialised with config: %s", self.get_embed_config()
        )

    def _load_existing_vector_store(self) -> tuple[VectorIndex, str | None]:
        """Load an existing vector store from a local folder or a GCS URI."""
        logger.info("Loading existing ClassifAI vector store from %s", self.db_dir)
        db_dir = self.db_dir

        if is_gcs_path(db_dir):
            self._downloaded_vector_store = download_vector_store_from_gcs(db_dir)
            db_dir = self._downloaded_vector_store.temp_dir.name

        if not has_persisted_vector_store(db_dir):
            raise FileNotFoundError(
                f"No existing vector store found in {self.db_dir}. "
                "Please ensure the directory contains metadata.json and vectors.parquet, "
                "or provide a valid index source file."
            )

        vector_store = self._backend.load(
            folder_path=db_dir,
            vectoriser=self.embeddings,
        )

        logger.info("Existing vector store loaded successfully from %s", self.db_dir)
        index_source_file = read_index_source_file(db_dir)
        return vector_store, index_source_file

    def _build_vector_store(self) -> VectorIndex:
        """Build a ClassifAI vector store from a CSV source file."""
        if not self.db_dir:
            raise ValueError("db_dir must be provided.")

        index_source_file = str(self.index_source_file)
        logger.info(
            "Building vector store in %s from source file %s.",
            self.db_dir,
            index_source_file,
        )

        if has_persisted_vectors_file(self.db_dir):
            logger.warning(
                "Existing vector store files found in %s. They will be overwritten.",
                self.db_dir,
            )

        if is_gcs_path(index_source_file):
            downloaded_file = download_one_file_from_gcs(index_source_file)
            index_source_file = downloaded_file.path

        vector_store = self._backend.build(
            file_name=index_source_file,
            vectoriser=self.embeddings,
            output_dir=self.db_dir,
        )

        write_index_source_file(self.db_dir, self.index_source_file)

        logger.info(
            "Vector store built successfully in %s with data from %s.",
            self.db_dir,
            self.index_source_file,
        )

        return vector_store

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
            embedding_model_name=self.embedding_model_name,
            db_dir=self.db_dir,
            k_matches=self.k_matches,
            index_source_file=self.index_source_file,
            index_size=self.index_size,
            status="ready",
        )
