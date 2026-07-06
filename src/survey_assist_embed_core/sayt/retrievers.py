"""Retriever implementations used by SAYT suggesters."""

# pylint: disable=too-few-public-methods, R0801

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from difflib import SequenceMatcher

from survey_assist_embed_core.sayt.core import CleanCorpus, Suggestion, take_with_ties
from survey_assist_embed_core.sayt.indexes import (
    DenseVectorIndex,
    build_ngram_index,
    build_semantic_index,
)

_FUZZY_PREFIX_MIN_RATIO = 0.75


@dataclass(frozen=True, slots=True)
class _PrefixIndex:
    """Precomputed prefix lookup structures for prefix matching."""

    token_index: dict[str, set[int]]


class PrefixRetriever:
    """Retrieve suggestions using exact, token, and fuzzy prefix matching."""

    def __init__(self, corpus: CleanCorpus, *, min_chars: int) -> None:
        """Initialise a prefix retriever.

        Args:
            corpus: Cleaned corpus to search.
            min_chars: Minimum query length required before retrieval runs.
        """
        self._min_chars = min_chars
        self.search_terms = [search_norm for search_norm, _ in corpus.rows]
        self.display_terms = [display_text for _, display_text in corpus.rows]
        self._index = self._build_index(self.search_terms)

    @staticmethod
    def _build_index(search_terms: list[str]) -> _PrefixIndex:
        """Precompute token-prefix lookup tables keyed by corpus row index."""
        token_index: dict[str, set[int]] = {}

        for row_ind, search_text in enumerate(search_terms):
            for token in search_text.split():
                for i in range(1, min(len(token), len(search_text)) + 1):
                    prefix_str = token[:i]
                    token_index.setdefault(prefix_str, set()).add(row_ind)

        return _PrefixIndex(token_index=token_index)

    def suggest_with_scores(
        self, q_norm: str, num_suggestions: int
    ) -> list[Suggestion]:
        """Return ranked prefix-based suggestions for a normalised query.

        Args:
            q_norm: Normalised query text.
            num_suggestions: Maximum number of scored results to return before
                tie expansion.

        Returns:
            Ranked ``Suggestion`` objects scored by prefix heuristics.
        """
        if len(q_norm) < self._min_chars:
            return []

        scores = [0.0] * len(self.search_terms)
        matched_inds: set[int] = set()

        left = bisect_left(self.search_terms, q_norm)
        right = bisect_right(self.search_terms, q_norm + "\uffff")
        for row_ind in range(left, right):
            scores[row_ind] += 3.0
            matched_inds.add(row_ind)

        for row_ind in self._index.token_index.get(q_norm, set()):
            scores[row_ind] += 2.5
            matched_inds.add(row_ind)

        for row_ind, search_norm in enumerate(self.search_terms):
            prefix = search_norm[: len(q_norm)]
            if not prefix:
                continue
            ratio = SequenceMatcher(a=q_norm, b=prefix).ratio()
            if ratio >= _FUZZY_PREFIX_MIN_RATIO:
                scores[row_ind] += 2.4 * ratio
                matched_inds.add(row_ind)

        suggestions = [
            Suggestion(
                display_text=self.display_terms[row_ind],
                score=scores[row_ind],
            )
            for row_ind in matched_inds
        ]
        return take_with_ties(suggestions, limit=num_suggestions)


class _DenseRetriever:
    """Shared cosine-similarity retrieval over an in-memory dense index."""

    _corpus: CleanCorpus
    _min_chars: int
    _index: DenseVectorIndex

    @classmethod
    def from_index(
        cls,
        corpus: CleanCorpus,
        *,
        min_chars: int,
        index: DenseVectorIndex,
    ) -> "_DenseRetriever":
        """Restore a dense retriever from an already-built dense index."""
        retriever = cls.__new__(cls)
        retriever._corpus = corpus
        retriever._min_chars = min_chars
        retriever._index = index
        return retriever

    def suggest_with_scores(
        self, q_norm: str, num_suggestions: int
    ) -> list[Suggestion]:
        """Return dense-vector suggestions after applying retriever-level gating."""
        if len(q_norm) < self._min_chars:
            return []
        return self._index.query(q_norm, num_suggestions)


class NgramRetriever(_DenseRetriever):
    """Retrieve suggestions using character n-gram similarity."""

    def __init__(
        self,
        corpus: CleanCorpus,
        *,
        n: int,
        max_df: float,
        min_chars: int,
    ) -> None:
        """Initialise a character n-gram retriever.

        Args:
            corpus: Cleaned corpus to search.
            n: Character n-gram size.
            max_df: Maximum document frequency passed to the n-gram vectoriser.
            min_chars: Minimum query length required before retrieval runs.
        """
        self._corpus = corpus
        self._min_chars = min_chars
        self._index = build_ngram_index(
            corpus=corpus,
            n=n,
            max_df=max_df,
        )


class SemanticRetriever(_DenseRetriever):
    """Retrieve suggestions using sentence-transformer embeddings."""

    def __init__(
        self,
        corpus: CleanCorpus,
        *,
        model: str,
        min_chars: int,
    ) -> None:
        """Initialise a semantic retriever.

        Args:
            corpus: Cleaned corpus to search.
            model: Sentence-transformer model name without the repository
                prefix.
            min_chars: Minimum query length required before retrieval runs.
        """
        self._corpus = corpus
        self._min_chars = min_chars
        self._index = build_semantic_index(
            corpus=corpus,
            model=model,
        )
