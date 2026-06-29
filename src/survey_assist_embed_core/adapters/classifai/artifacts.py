"""Helpers for the persisted ClassifAI vector-store layout."""

import json
import os

METADATA_FILE_NAME = "metadata.json"
VECTORS_FILE_NAME = "vectors.parquet"
INDEX_SOURCE_FILE_KEY = "index_source_file"
EMBEDDING_MODEL_NAME_KEY = "embedding_model_name"


def _metadata_path(folder_path: str) -> str:
    """Return the metadata file path for a persisted store folder."""
    return os.path.join(folder_path, METADATA_FILE_NAME)


def _vectors_path(folder_path: str) -> str:
    """Return the vectors file path for a persisted store folder."""
    return os.path.join(folder_path, VECTORS_FILE_NAME)


def _read_metadata(folder_path: str) -> dict[str, str]:
    """Read persisted metadata for a vector store folder if present."""
    metadata_path = _metadata_path(folder_path)
    if not os.path.exists(metadata_path):
        return {}

    with open(metadata_path, encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _write_metadata(folder_path: str, metadata: dict[str, str]) -> None:
    """Write persisted metadata for a vector store folder."""
    metadata_path = _metadata_path(folder_path)
    with open(metadata_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj)


def _write_metadata_value(*, folder_path: str, key: str, value: str | None) -> None:
    """Write or update a single persisted metadata value."""
    metadata = _read_metadata(folder_path)
    metadata[key] = str(value)
    _write_metadata(folder_path, metadata)


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


def read_index_source_file(*, folder_path: str) -> str | None:
    """Read the original source-file path from persisted metadata."""
    metadata = _read_metadata(folder_path)
    return metadata.get(INDEX_SOURCE_FILE_KEY)


def write_index_source_file(*, folder_path: str, index_source_file: str | None) -> None:
    """Write or update the original source-file path in persisted metadata."""
    _write_metadata_value(
        folder_path=folder_path,
        key=INDEX_SOURCE_FILE_KEY,
        value=index_source_file,
    )


def read_embedding_model_name(*, folder_path: str) -> str | None:
    """Read the embedding model name from persisted metadata."""
    metadata = _read_metadata(folder_path)
    return metadata.get(EMBEDDING_MODEL_NAME_KEY)


def write_embedding_model_name(
    *, folder_path: str, embedding_model_name: str | None
) -> None:
    """Write or update the embedding model name in persisted metadata."""
    _write_metadata_value(
        folder_path=folder_path,
        key=EMBEDDING_MODEL_NAME_KEY,
        value=embedding_model_name,
    )
