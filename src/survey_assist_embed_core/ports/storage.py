"""Ports for source-file and persisted-store path resolution."""

from typing import Protocol


class Storage(Protocol):
    """Protocol for resolving local and remote storage locations."""

    def resolve_store_path(self, *, path: str) -> str:
        """Return a local folder path for a persisted vector store location."""

    def resolve_source_file(self, *, path: str) -> str:
        """Return a local file path for a source-file location."""
