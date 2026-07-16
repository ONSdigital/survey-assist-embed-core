# pylint: disable=too-few-public-methods

"""Dense index construction helpers for SAYT retrievers."""

import csv
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import classifai.indexers.main as classifai_indexers_main
import numpy as np
from classifai.indexers import VectorStore, VectorStoreSearchInput
from classifai.vectorisers import VectoriserBase
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from survey_assist_utils import get_logger

from survey_assist_embed_core.adapters.classifai.vector_backend import (
    resolve_model_name,
)
from survey_assist_embed_core.adapters.classifai.vectoriser import (
    build_vectoriser,
    normalise_vectors,
)
from survey_assist_embed_core.sayt.core import CleanCorpus, Suggestion, take_with_ties

logger = get_logger(__name__)


def _silent_tqdm(iterable, **_kwargs):
    """Pass through iterables unchanged to suppress ClassifAI progress bars."""
    return iterable


@contextmanager
def _silence_classifai_tqdm():
    """Temporarily replace ClassifAI's tqdm import with a no-op wrapper."""
    previous_tqdm = classifai_indexers_main.tqdm
    classifai_indexers_main.tqdm = _silent_tqdm
    try:
        yield
    finally:
        classifai_indexers_main.tqdm = previous_tqdm


def _derive_num_retrieved_based_on_duplication(
    n_suggestions: int, max_duplication: int, corpus_size: int
) -> int:
    """Derive dense retrieval number of candidates from requested size and corpus duplication.

    The number of candidates is intentionally dampened using ``log2(max_duplication) + 1``
    rather than scaling linearly with ``max_duplication``. This keeps candidate
    growth sub-linear when many rows share the same display text, balancing
    recall against dense query cost.

    Args:
        n_suggestions: Requested number of final suggestions.
        max_duplication: Maximum count of any display text in the corpus.
        corpus_size: Total number of indexed rows.

    Returns:
        Candidate count for vector-store search, capped at corpus size.
    """
    if n_suggestions < 1 or corpus_size < 1:
        return 0

    dampen_max_duplication = int(np.log2(max(max_duplication, 1)) + 1)

    out = min(corpus_size, n_suggestions * dampen_max_duplication)
    return out


@dataclass(frozen=True, slots=True)
class DenseVectorIndex:
    """Wrap a ClassifAI vector store for query-time dense retrieval."""

    _vector_store: VectorStore
    _num_vectors: int
    _corpus: CleanCorpus
    _max_duplication: int = 1

    @classmethod
    def from_corpus(
        cls,
        *,
        corpus: CleanCorpus,
        vectoriser: VectoriserBase,
        output_dir: str | os.PathLike[str] | None = None,
        overwrite: bool = True,
    ) -> "DenseVectorIndex":
        """Build a dense index from a cleaned corpus.

        Args:
            corpus: Cleaned corpus whose normalised search text should be
                indexed.
            vectoriser: Vectoriser used to embed corpus rows and future queries.
            output_dir: Optional persistent filespace directory for the
                underlying ClassifAI vector store. When omitted, a temporary
                directory is used.
            overwrite: Whether to allow ClassifAI to replace an existing
                filespace when ``output_dir`` is provided.

        Returns:
            A ``DenseVectorIndex`` backed by ClassifAI's ``VectorStore``.
        """
        with tempfile.TemporaryDirectory(prefix="sayt_") as temp_dir:
            csv_path = os.path.join(temp_dir, "corpus.csv")
            classifai_output_dir = (
                os.fspath(output_dir)
                if output_dir is not None
                else os.path.join(temp_dir, "vector_store")
            )

            cls._write_corpus_csv(corpus, csv_path)

            with _silence_classifai_tqdm():
                vector_store = VectorStore(
                    file_name=csv_path,
                    data_type="csv",
                    vectoriser=vectoriser,
                    batch_size=64,
                    output_dir=classifai_output_dir,
                    overwrite=overwrite,
                    hooks=None,
                )

        return cls(
            _vector_store=vector_store,
            _num_vectors=int(vector_store.num_vectors or 0),
            _corpus=corpus,
            _max_duplication=max(corpus.display_text_value_counts.values(), default=1),
        )

    @classmethod
    def from_filespace(
        cls,
        *,
        corpus: CleanCorpus,
        folder_path: str | os.PathLike[str],
        vectoriser: VectoriserBase,
    ) -> "DenseVectorIndex":
        """Load a dense index from a persisted ClassifAI filespace.

        Args:
            corpus: Cleaned corpus whose row metadata should back query results.
            folder_path: Filesystem directory containing ``metadata.json`` and
                ``vectors.parquet``.
            vectoriser: Vectoriser used to embed future query text.

        Returns:
            A ``DenseVectorIndex`` backed by a loaded ``VectorStore``.
        """
        with _silence_classifai_tqdm():
            vector_store = VectorStore.from_filespace(
                folder_path=os.fspath(folder_path),
                vectoriser=vectoriser,
                hooks=None,
            )

        return cls(
            _vector_store=vector_store,
            _num_vectors=int(vector_store.num_vectors or 0),
            _corpus=corpus,
            _max_duplication=max(corpus.display_text_value_counts.values(), default=1),
        )

    @staticmethod
    def _write_corpus_csv(
        corpus: CleanCorpus,
        csv_path: str | os.PathLike[str],
    ) -> None:
        """Write the display-label and search-text schema expected by ClassifAI."""
        csv_file = Path(csv_path)
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["label", "text"])
            writer.writeheader()
            writer.writerows(
                {"label": display_text, "text": search_text}
                for search_text, display_text in corpus.rows
            )

    def query(self, q_norm: str, num_suggestions: int) -> list[Suggestion]:
        """Query the dense index with a normalised string.

        Args:
            q_norm: Normalised query text.
            num_suggestions: Maximum number of scored suggestions to return before
                tie expansion.

        Returns:
            Ranked ``Suggestion`` objects from the dense vector store.

        Notes:
            - Dense retrieval first fetches more than ``num_suggestions`` using
              ``_derive_num_retrieved_based_on_duplication``.
            - The widened candidate list is then ranked and trimmed with
              ``take_with_ties`` to preserve cutoff ties.
            - If the widened candidate list is still too small to satisfy ``num_suggestions``,
              the retrieval is repeated with a larger candidate count until either
              the corpus is exhausted or enough candidates are found.
        """
        if self._num_vectors < 1 or num_suggestions < 1:
            return []

        start_time = time.time()

        search_input = VectorStoreSearchInput({"id": ["q1"], "query": [q_norm]})
        num_results = _derive_num_retrieved_based_on_duplication(
            num_suggestions, self._max_duplication, self._num_vectors
        )

        while True:
            with _silence_classifai_tqdm():
                results = self._vector_store.search(search_input, n_results=num_results)

            labels = results["doc_label"].tolist()
            scores = results["score"].tolist()
            suggestions = [
                Suggestion(display_text=label, score=score)
                for label, score in zip(labels, scores, strict=True)
            ]
            out = take_with_ties(suggestions, limit=num_suggestions)

            elapsed_time = time.time() - start_time
            logger.debug(
                "Dense index query time (low level)",
                query_time_ms=elapsed_time * 1000,
                num_sem_results_requested=num_results,
                num_sem_results_returned=len(suggestions),
                num_suggestions_requested=num_suggestions,
                num_suggestions_returned=len(out),
            )

            if len(out) < num_suggestions and num_results < self._num_vectors:
                num_results = min(self._num_vectors, num_results * 2)
            else:
                return out


class _CharNgramVectoriser(VectoriserBase):
    """CountVectorizer char_wb n-gram vectoriser with unit-length outputs."""

    def __init__(self, corpus: list[str], *, n: int, max_df: float) -> None:
        """Fit a character n-gram vectoriser on the normalised corpus."""
        self._vectoriser = CountVectorizer(
            analyzer="char_wb",
            ngram_range=(n, n),
            max_df=max_df,
        )
        self._vectoriser.fit(corpus)

    def transform(self, texts: str | list[str]) -> np.ndarray:
        """Return unit-length dense n-gram vectors for the provided texts."""
        if isinstance(texts, str):
            texts = [texts]
        matrix = cast(csr_matrix, self._vectoriser.transform(texts))
        vectors = matrix.toarray().astype(float, copy=False)

        return normalise_vectors(vectors)


def build_ngram_index(
    corpus: CleanCorpus,
    *,
    n: int,
    max_df: float,
    output_dir: str | os.PathLike[str] | None = None,
    overwrite: bool = True,
) -> DenseVectorIndex:
    """Build a dense index backed by character n-gram vectors.

    Args:
        corpus: Cleaned corpus to index.
        n: Character n-gram size.
        max_df: Maximum document frequency passed to ``CountVectorizer``.
        output_dir: Optional persistent filespace directory for the generated
            vector store.
        overwrite: Whether to allow ClassifAI to replace an existing filespace
            when ``output_dir`` is provided.

    Returns:
        A dense index using character n-gram embeddings.
    """
    return DenseVectorIndex.from_corpus(
        corpus=corpus,
        vectoriser=_CharNgramVectoriser(
            [search for search, _ in corpus.rows],
            n=n,
            max_df=max_df,
        ),
        output_dir=output_dir,
        overwrite=overwrite,
    )


def load_ngram_index(
    corpus: CleanCorpus,
    *,
    n: int,
    max_df: float,
    folder_path: str | os.PathLike[str],
) -> DenseVectorIndex:
    """Load a persisted dense index backed by character n-gram vectors."""
    return DenseVectorIndex.from_filespace(
        corpus=corpus,
        folder_path=folder_path,
        vectoriser=_CharNgramVectoriser(
            [search for search, _ in corpus.rows],
            n=n,
            max_df=max_df,
        ),
    )


def build_semantic_index(
    corpus: CleanCorpus,
    *,
    model: str,
    vectoriser_class: str | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    overwrite: bool = True,
) -> DenseVectorIndex:
    """Build a dense index backed by sentence-transformer embeddings.

    Args:
        corpus: Cleaned corpus to index.
        model: Sentence-transformer model name without the repository prefix.
        vectoriser_class: Optional semantic vectoriser class to use ("ONNX" or "HF").
            If not provided, the default ONNX vectoriser is used.
        output_dir: Optional persistent filespace directory for the generated
            vector store.
        overwrite: Whether to allow ClassifAI to replace an existing filespace
            when ``output_dir`` is provided.

    Returns:
        A dense index using semantic embeddings.
    """
    semantic_model = resolve_model_name(model)
    semantic_vectoriser = build_vectoriser(
        embedding_model_name=semantic_model,
        vectoriser_class=vectoriser_class,
    )

    return DenseVectorIndex.from_corpus(
        corpus=corpus,
        vectoriser=semantic_vectoriser,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def load_semantic_index(
    corpus: CleanCorpus,
    *,
    model: str,
    vectoriser_class: str | None = None,
    folder_path: str | os.PathLike[str],
) -> DenseVectorIndex:
    """Load a persisted dense index backed by semantic embeddings."""
    semantic_model = resolve_model_name(model)
    semantic_vectoriser = build_vectoriser(
        embedding_model_name=semantic_model,
        vectoriser_class=vectoriser_class,
    )

    return DenseVectorIndex.from_filespace(
        corpus=corpus,
        folder_path=folder_path,
        vectoriser=semantic_vectoriser,
    )
