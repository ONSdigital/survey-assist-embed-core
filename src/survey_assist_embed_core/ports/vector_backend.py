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
        """Return backend search rows for a query."""


class VectorBackend(Protocol):
    """Protocol for build/load operations against a vector backend."""

    @property
    def config(self) -> VectorBackendConfig:
        """Return typed backend metadata for diagnostics and status output."""

    def has_persisted_store(self, *, folder_path: str) -> bool:
        """Return whether persisted backend artifacts already exist."""

    def load(self, *, folder_path: str) -> tuple[VectorIndex, str | None]:
        """Load a vector index and recorded source path from a persisted folder."""

    def build(
        self,
        *,
        file_name: str,
        output_dir: str,
        index_source_file: str | None,
    ) -> None:
        """Build persisted backend artifacts from a prepared source file."""
