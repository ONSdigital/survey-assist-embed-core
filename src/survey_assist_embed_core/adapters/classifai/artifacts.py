"""Helpers for the persisted ClassifAI vector-store layout."""

import json
import os

METADATA_FILE_NAME = "metadata.json"
VECTORS_FILE_NAME = "vectors.parquet"
INDEX_SOURCE_FILE_KEY = "index_source_file"


def has_persisted_vector_store(folder_path: str) -> bool:
    """Return whether the expected persisted ClassifAI files are present."""
    metadata_path = os.path.join(folder_path, METADATA_FILE_NAME)
    vectors_path = os.path.join(folder_path, VECTORS_FILE_NAME)
    return (
        os.path.isdir(folder_path)
        and os.path.exists(metadata_path)
        and os.path.exists(vectors_path)
    )


def has_persisted_vectors_file(folder_path: str) -> bool:
    """Return whether the persisted vectors file already exists."""
    return os.path.exists(os.path.join(folder_path, VECTORS_FILE_NAME))


def read_index_source_file(folder_path: str) -> str | None:
    """Read the original source-file path from persisted metadata."""
    metadata_path = os.path.join(folder_path, METADATA_FILE_NAME)
    with open(metadata_path, encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    return metadata.get(INDEX_SOURCE_FILE_KEY, None)


def write_index_source_file(folder_path: str, index_source_file: str | None) -> None:
    """Write or update the original source-file path in persisted metadata."""
    metadata_path = os.path.join(folder_path, METADATA_FILE_NAME)
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as file_obj:
            metadata = json.load(file_obj)
    else:
        metadata = {}

    metadata[INDEX_SOURCE_FILE_KEY] = str(index_source_file)
    with open(metadata_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj)
