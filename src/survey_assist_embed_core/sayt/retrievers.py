"""Retriever implementations used by SAYT suggesters."""

# pylint: disable=too-few-public-methods, R0801

from dataclasses import dataclass, field

from rapidfuzz import fuzz

from survey_assist_embed_core.sayt.core import CleanCorpus, Suggestion, take_with_ties
from survey_assist_embed_core.sayt.indexes import (
    DenseVectorIndex,
    build_ngram_index,
    build_semantic_index,
)

_FUZZY_PREFIX_MIN_RATIO = 0.75


@dataclass(slots=True)
class _PrefixTrieNode:
    """Trie node storing child links and matching row indices for a prefix."""

    children: dict[str, "_PrefixTrieNode"] = field(default_factory=dict)
    row_inds: set[int] = field(default_factory=set)


def _insert_prefix(root: _PrefixTrieNode, text: str, row_ind: int) -> None:
    """Insert all prefixes of ``text`` into ``root`` for ``row_ind``."""
    node = root
    for char in text:
        node = node.children.setdefault(char, _PrefixTrieNode())
        node.row_inds.add(row_ind)


def _lookup_prefix(root: _PrefixTrieNode, prefix: str) -> set[int]:
    """Return row indices that match ``prefix`` within the trie."""
    node: _PrefixTrieNode | None = root
    for char in prefix:
        if node is None:
            return set()
        node = node.children.get(char)
    if node is None:
        return set()
    return node.row_inds


@dataclass(frozen=True, slots=True)
class _PrefixIndex:
    """Precomputed prefix lookup structures for prefix matching."""

    search_trie: _PrefixTrieNode
    token_trie: _PrefixTrieNode


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
        """Precompute search-prefix and token-prefix tries keyed by row index."""
        search_trie = _PrefixTrieNode()
        token_trie = _PrefixTrieNode()

        for row_ind, search_text in enumerate(search_terms):
            _insert_prefix(search_trie, search_text, row_ind)
            for token in search_text.split():
                _insert_prefix(token_trie, token, row_ind)

        return _PrefixIndex(search_trie=search_trie, token_trie=token_trie)

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

        for row_ind in _lookup_prefix(self._index.search_trie, q_norm):
            scores[row_ind] += 3.0
            matched_inds.add(row_ind)

        for row_ind in _lookup_prefix(self._index.token_trie, q_norm):
            scores[row_ind] += 2.5
            matched_inds.add(row_ind)

        for row_ind, search_norm in enumerate(self.search_terms):
            prefix = search_norm[: len(q_norm)]
            if not prefix:
                continue
            ratio = fuzz.ratio(q_norm, prefix) / 100.0
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
        vectoriser_class: str | None = None,
        min_chars: int,
    ) -> None:
        """Initialise a semantic retriever.

        Args:
            corpus: Cleaned corpus to search.
            model: Sentence-transformer model name without the repository
                prefix.
            vectoriser_class: Optional semantic vectoriser class or alias.
            min_chars: Minimum query length required before retrieval runs.
        """
        self._corpus = corpus
        self._min_chars = min_chars
        self._index = build_semantic_index(
            corpus=corpus,
            model=model,
            vectoriser_class=vectoriser_class,
        )
