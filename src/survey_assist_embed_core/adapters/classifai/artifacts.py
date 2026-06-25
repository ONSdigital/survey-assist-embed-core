"""Helpers for the persisted ClassifAI vector-store layout."""

import json
import os

from survey_assist_embed_core.ports import ArtifactStore

METADATA_FILE_NAME = "metadata.json"
VECTORS_FILE_NAME = "vectors.parquet"
INDEX_SOURCE_FILE_KEY = "index_source_file"


class ClassifaiArtifactStore(ArtifactStore):
    """Persisted artifact adapter for the ClassifAI vector-store layout."""

    def __init__(
        self,
        *,
        metadata_file_name: str = METADATA_FILE_NAME,
        vectors_file_name: str = VECTORS_FILE_NAME,
        index_source_file_key: str = INDEX_SOURCE_FILE_KEY,
    ) -> None:
        """Store the ClassifAI persisted-layout policy for later file operations."""
        self._metadata_file_name = metadata_file_name
        self._vectors_file_name = vectors_file_name
        self._index_source_file_key = index_source_file_key

    def _metadata_path(self, folder_path: str) -> str:
        """Return the metadata file path for a persisted store folder."""
        return os.path.join(folder_path, self._metadata_file_name)

    def _vectors_path(self, folder_path: str) -> str:
        """Return the vectors file path for a persisted store folder."""
        return os.path.join(folder_path, self._vectors_file_name)

    def _has_persisted_vector_store(self, folder_path: str) -> bool:
        """Return whether the expected persisted ClassifAI files are present."""
        metadata_path = self._metadata_path(folder_path)
        vectors_path = self._vectors_path(folder_path)
        return (
            os.path.isdir(folder_path)
            and os.path.exists(metadata_path)
            and os.path.exists(vectors_path)
        )

    def ensure_persisted_vector_store(self, *, folder_path: str) -> None:
        """Raise when the folder is missing the persisted files for this layout."""
        if self._has_persisted_vector_store(folder_path):
            return

        required_artifacts = ", ".join(
            (self._metadata_file_name, self._vectors_file_name)
        )
        raise FileNotFoundError(
            f"No persisted vector store found in {folder_path}. "
            f"Required persisted artifacts: {required_artifacts}."
        )

    def has_persisted_vectors_file(self, *, folder_path: str) -> bool:
        """Return whether the persisted vectors file already exists."""
        return os.path.exists(self._vectors_path(folder_path))

    def read_index_source_file(self, *, folder_path: str) -> str | None:
        """Read the original source-file path from persisted metadata."""
        metadata_path = self._metadata_path(folder_path)
        with open(metadata_path, encoding="utf-8") as file_obj:
            metadata = json.load(file_obj)
        return metadata.get(self._index_source_file_key, None)

    def write_index_source_file(
        self, *, folder_path: str, index_source_file: str | None
    ) -> None:
        """Write or update the original source-file path in persisted metadata."""
        metadata_path = self._metadata_path(folder_path)
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as file_obj:
                metadata = json.load(file_obj)
        else:
            metadata = {}

        metadata[self._index_source_file_key] = str(index_source_file)
        with open(metadata_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj)
