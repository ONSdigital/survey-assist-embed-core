"""Embedding and retrieval helpers backed by a ClassifAI vector store."""

# pylint: disable=too-many-instance-attributes

from __future__ import annotations

import json
import logging
import os
from typing import Protocol, cast

import numpy as np
from autocorrect import Speller
from classifai.indexers import VectorStore, VectorStoreSearchInput
from classifai.vectorisers import HuggingFaceVectoriser

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

DEFAULT_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_DB_DIR = "vector_store"
DEFAULT_K_MATCHES = 20

logger = logging.getLogger(__name__)


class _SearchResultsWithToDicts(Protocol):  # pylint: disable=too-few-public-methods
    def to_dicts(self) -> list[dict[str, object]]:
        """Return rows as dictionaries."""


class _SearchResultsWithToDict(Protocol):  # pylint: disable=too-few-public-methods
    def to_dict(self, orient: str = "records") -> list[dict[str, object]]:
        """Return rows as dictionaries."""


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
    """Handle embedding operations for a ClassifAI vector store."""

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        db_dir: str = DEFAULT_DB_DIR,
        k_matches: int = DEFAULT_K_MATCHES,
        index_source_file: str | None = None,
    ):
        """Initialise the handler for an existing or newly built vector store."""
        self.embedding_model_name = embedding_model_name
        self.k_matches = k_matches
        self.db_dir = db_dir
        self.index_source_file = index_source_file

        self.embeddings: HuggingFaceVectoriser = ChromaDBesqueHFVectoriser(
            model_name=f"sentence-transformers/{embedding_model_name}"
        )
        logger.info("Using embedding model: %s", embedding_model_name)

        self.spell = Speller()

        self._downloaded_vector_store: DownloadedVectorStore | None = None
        self.vector_store: VectorStore
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

    def _load_existing_vector_store(self) -> tuple[VectorStore, str | None]:
        """Load an existing vector store from a local folder or a GCS URI."""
        logger.info("Loading existing ClassifAI vector store from %s", self.db_dir)
        db_dir = self.db_dir

        if is_gcs_path(db_dir):
            self._downloaded_vector_store = download_vector_store_from_gcs(db_dir)
            db_dir = self._downloaded_vector_store.temp_dir.name

        metadata_path = os.path.join(db_dir, "metadata.json")
        vectors_path = os.path.join(db_dir, "vectors.parquet")

        has_existing_store = (
            os.path.isdir(db_dir)
            and os.path.exists(metadata_path)
            and os.path.exists(vectors_path)
        )

        if not has_existing_store:
            raise FileNotFoundError(
                f"No existing vector store found in {self.db_dir}. "
                "Please ensure the directory contains metadata.json and vectors.parquet, "
                "or provide a valid index source file."
            )

        vector_store = VectorStore.from_filespace(
            folder_path=db_dir,
            vectoriser=self.embeddings,
            hooks=None,
        )

        logger.info("Existing vector store loaded successfully from %s", self.db_dir)
        with open(metadata_path, encoding="utf-8") as file_obj:
            metadata = json.load(file_obj)
        index_source_file = metadata.get("index_source_file", None)
        return vector_store, index_source_file

    def _build_vector_store(self) -> VectorStore:
        """Build a ClassifAI vector store from a CSV source file."""
        if not self.db_dir:
            raise ValueError("db_dir must be provided.")

        index_source_file = str(self.index_source_file)
        logger.info(
            "Building vector store in %s from source file %s.",
            self.db_dir,
            index_source_file,
        )

        if os.path.exists(os.path.join(self.db_dir, "vectors.parquet")):
            logger.warning(
                "Existing vector store files found in %s. They will be overwritten.",
                self.db_dir,
            )

        if is_gcs_path(index_source_file):
            downloaded_file = download_one_file_from_gcs(index_source_file)
            index_source_file = downloaded_file.path

        vector_store = VectorStore(
            file_name=index_source_file,
            data_type="csv",
            vectoriser=self.embeddings,
            batch_size=8,
            meta_data=None,
            output_dir=self.db_dir,
            overwrite=True,
            hooks=None,
        )

        metadata_path = os.path.join(self.db_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as file_obj:
                metadata = json.load(file_obj)
        else:
            metadata = {}
        metadata["index_source_file"] = str(self.index_source_file)
        with open(metadata_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj)

        logger.info(
            "Vector store built successfully in %s with data from %s.",
            self.db_dir,
            self.index_source_file,
        )

        return vector_store

    def search_index(self, query: str) -> SearchIndexResponse:
        """Return the nearest index entries for a query string."""
        search_input = VectorStoreSearchInput({"id": ["q1"], "query": [query]})

        n_results = min(self.index_size, self.k_matches)
        results = self.vector_store.search(search_input, n_results=n_results)

        if hasattr(results, "to_dicts"):
            rows = cast(_SearchResultsWithToDicts, results).to_dicts()
        else:
            rows = cast(_SearchResultsWithToDict, results).to_dict(orient="records")

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
