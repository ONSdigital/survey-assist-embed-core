"""Ports for persisted vector-store artifact layouts."""

from typing import Protocol


class ArtifactStore(Protocol):
    """Protocol for persisted vector-store artifact operations."""

    def ensure_persisted_vector_store(self, *, folder_path: str) -> None:
        """Raise if the persisted store folder does not match this artifact layout."""

    def has_persisted_vectors_file(self, *, folder_path: str) -> bool:
        """Return whether the persisted vectors file is present."""

    def read_index_source_file(self, *, folder_path: str) -> str | None:
        """Read the source-file path recorded in persisted metadata."""

    def write_index_source_file(
        self, *, folder_path: str, index_source_file: str | None
    ) -> None:
        """Write the source-file path into persisted metadata."""
