"""Embedding and retrieval helpers backed by pluggable vector-store adapters."""

# pylint: disable=too-many-instance-attributes

import logging
from typing import cast

import numpy as np
from autocorrect import Speller
from classifai.vectorisers import HuggingFaceVectoriser

from survey_assist_embed_core.adapters.classifai import (
    ClassifaiArtifactStore,
    ClassifaiVectorBackend,
)
from survey_assist_embed_core.adapters.storage import LocalGcsStorage
from survey_assist_embed_core.models import (
    EmbeddingStatus,
    SearchIndexItem,
    SearchIndexResponse,
)
from survey_assist_embed_core.ports import (
    ArtifactStore,
    Storage,
    VectorBackend,
    VectorIndex,
)

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

    def __init__(  # noqa: PLR0913 - constructor wires explicit handler dependencies.
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        db_dir: str = DEFAULT_DB_DIR,
        k_matches: int = DEFAULT_K_MATCHES,
        index_source_file: str | None = None,
        backend: VectorBackend | None = None,
        artifact_store: ArtifactStore | None = None,
        storage: Storage | None = None,
    ):
        """Initialise the handler for an existing or newly built vector store."""
        self.embedding_model_name = embedding_model_name
        self.k_matches = k_matches
        self.db_dir = db_dir
        self.index_source_file = index_source_file
        self._backend = backend if backend is not None else ClassifaiVectorBackend()
        self._artifact_store = (
            artifact_store if artifact_store is not None else ClassifaiArtifactStore()
        )
        self._storage = storage if storage is not None else LocalGcsStorage()

        self.embeddings: HuggingFaceVectoriser = ChromaDBesqueHFVectoriser(
            model_name=f"sentence-transformers/{embedding_model_name}"
        )
        logger.info("Using embedding model: %s", embedding_model_name)

        self.spell = Speller()

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
        logger.info("Loading existing vector store from %s", self.db_dir)
        db_dir = self._storage.resolve_store_path(path=self.db_dir)

        try:
            self._artifact_store.ensure_persisted_vector_store(folder_path=db_dir)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc} Or provide a valid index source file."
            ) from exc

        vector_store = self._backend.load(
            folder_path=db_dir,
            vectoriser=self.embeddings,
        )

        logger.info("Existing vector store loaded successfully from %s", self.db_dir)
        index_source_file = self._artifact_store.read_index_source_file(
            folder_path=db_dir
        )
        return vector_store, index_source_file

    def _build_vector_store(self) -> VectorIndex:
        """Build a vector store from a CSV source file."""
        if not self.db_dir:
            raise ValueError("db_dir must be provided.")

        index_source_file = str(self.index_source_file)
        logger.info(
            "Building vector store in %s from source file %s.",
            self.db_dir,
            index_source_file,
        )

        if self._artifact_store.has_persisted_vectors_file(folder_path=self.db_dir):
            logger.warning(
                "Existing vector store files found in %s. They will be overwritten.",
                self.db_dir,
            )

        index_source_file = self._storage.resolve_source_file(path=index_source_file)

        vector_store = self._backend.build(
            file_name=index_source_file,
            vectoriser=self.embeddings,
            output_dir=self.db_dir,
        )

        self._artifact_store.write_index_source_file(
            folder_path=self.db_dir,
            index_source_file=self.index_source_file,
        )

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
