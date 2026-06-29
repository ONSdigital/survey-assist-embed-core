"""Helpers for the persisted ClassifAI vector-store layout."""

import json
import os

METADATA_FILE_NAME = "metadata.json"
VECTORS_FILE_NAME = "vectors.parquet"
INDEX_SOURCE_FILE_KEY = "index_source_file"


def _metadata_path(folder_path: str) -> str:
    """Return the metadata file path for a persisted store folder."""
    return os.path.join(folder_path, METADATA_FILE_NAME)


def _vectors_path(folder_path: str) -> str:
    """Return the vectors file path for a persisted store folder."""
    return os.path.join(folder_path, VECTORS_FILE_NAME)


def _has_persisted_vector_store(folder_path: str) -> bool:
    """Return whether the expected persisted ClassifAI files are present."""
    metadata_path = _metadata_path(folder_path)
    vectors_path = _vectors_path(folder_path)
    return (
        os.path.isdir(folder_path)
        and os.path.exists(metadata_path)
        and os.path.exists(vectors_path)
    )


def ensure_persisted_vector_store(*, folder_path: str) -> None:
    """Raise when the folder is missing the persisted files for this layout."""
    if _has_persisted_vector_store(folder_path):
        return

    required_artifacts = ", ".join((METADATA_FILE_NAME, VECTORS_FILE_NAME))
    raise FileNotFoundError(
        f"No persisted vector store found in {folder_path}. "
        f"Required persisted artifacts: {required_artifacts}."
    )


def has_persisted_vectors_file(*, folder_path: str) -> bool:
    """Return whether the persisted vectors file already exists."""
    return os.path.exists(_vectors_path(folder_path))


def read_index_source_file(*, folder_path: str) -> str | None:
    """Read the original source-file path from persisted metadata."""
    metadata_path = _metadata_path(folder_path)
    with open(metadata_path, encoding="utf-8") as file_obj:
        metadata: dict = json.load(file_obj)
    return metadata.get(INDEX_SOURCE_FILE_KEY)


def write_index_source_file(*, folder_path: str, index_source_file: str | None) -> None:
    """Write or update the original source-file path in persisted metadata."""
    metadata_path = _metadata_path(folder_path)
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as file_obj:
            metadata = json.load(file_obj)
    else:
        metadata = {}

    metadata[INDEX_SOURCE_FILE_KEY] = str(index_source_file)
    with open(metadata_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj)
