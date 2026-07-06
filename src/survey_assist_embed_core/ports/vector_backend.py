"""Ports for vector-store backends."""

from collections.abc import Mapping
from typing import Protocol

from survey_assist_embed_core.models.status import VectorBackendConfig

type SearchRow = Mapping[str, object]


class VectorIndex(Protocol):
    """Protocol for a loaded vector index."""

    @property
    def num_vectors(self) -> int | None:
        """Return the number of vectors in the index if known."""

    def search(self, query: str, *, limit: int) -> list[SearchRow]:
        """Return backend search rows for a single query.

        Args:
            query: Search text to submit to the backend.
            limit: Maximum number of rows to return.

        Returns:
            Backend-specific rows describing the ranked matches.
        """

    def search_many(self, queries: list[str], *, limit: int) -> list[list[SearchRow]]:
        """Return backend search rows for multiple queries in input order.

        Args:
            queries: Search strings to submit in a single backend batch.
            limit: Maximum number of rows to return for each query.

        Returns:
            Ranked backend rows grouped in the same order as ``queries``.
        """


class VectorBackend(Protocol):
    """Protocol for runtime operations against a vector backend."""

    @property
    def config(self) -> VectorBackendConfig:
        """Return typed backend metadata for diagnostics and status output."""

    def load(self, *, folder_path: str) -> tuple[VectorIndex, str | None]:
        """Load a vector index and recorded source path from a persisted folder.

        Args:
            folder_path: Local folder containing persisted vector-store
                artifacts.

        Returns:
            A tuple of the loaded vector index and the recorded source-file
            path, if available.
        """
