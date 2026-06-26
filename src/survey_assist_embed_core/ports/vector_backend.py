"""Ports for vector-store backends."""

from collections.abc import Mapping
from typing import Protocol

type SearchRow = Mapping[str, object]


class VectorIndex(Protocol):
    """Protocol for a loaded vector index."""

    @property
    def num_vectors(self) -> int | None:
        """Return the number of vectors in the index if known."""

    def search(self, query: str, *, limit: int) -> list[SearchRow]:
        """Return backend search rows for a query."""


class VectorBackend(Protocol):
    """Protocol for build/load operations against a vector backend."""

    def load(self, *, folder_path: str, embedding_model_name: str) -> VectorIndex:
        """Load a vector index from a persisted folder."""

    def build(
        self, *, file_name: str, embedding_model_name: str, output_dir: str
    ) -> VectorIndex:
        """Build a vector index from a prepared source file."""
