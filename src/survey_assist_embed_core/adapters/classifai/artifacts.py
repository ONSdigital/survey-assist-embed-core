"""Helpers for the persisted ClassifAI vector-store layout."""

import json
import os

METADATA_FILE_NAME = "metadata.json"
VECTORS_FILE_NAME = "vectors.parquet"
INDEX_SOURCE_FILE_KEY = "index_source_file"
EMBEDDING_MODEL_NAME_KEY = "embedding_model_name"


def write_vector_store_metadata(
    *,
    folder_path: str,
    index_source_file: str | None,
    embedding_model_name: str | None,
) -> None:
    """Write project metadata into a persisted vector-store folder.

    classifai rewrites ``metadata.json`` when it builds a vector store, so this
    helper only adds the project-specific keys after verifying that those keys
    are not already present.

    Args:
        folder_path: Folder that contains the persisted vector-store artifacts.
        index_source_file: Original source-file path to record in the metadata.
        embedding_model_name: Embedding model identifier to record.

    Raises:
        ValueError: If the metadata file already contains one of the reserved
            project keys.
    """
    our_keys = {INDEX_SOURCE_FILE_KEY, EMBEDDING_MODEL_NAME_KEY}
    existing = set(_read_metadata(folder_path).keys())
    collisions = our_keys & existing
    if collisions:
        raise ValueError(
            f"Metadata keys {sorted(collisions)} are already present in "
            f"{_metadata_path(folder_path)!r}."
        )

    metadata = _read_metadata(folder_path)
    metadata[INDEX_SOURCE_FILE_KEY] = str(index_source_file)
    metadata[EMBEDDING_MODEL_NAME_KEY] = str(embedding_model_name)
    _write_metadata(folder_path, metadata)


def ensure_persisted_vector_store(*, folder_path: str) -> None:
    """Validate that a folder contains the expected persisted artifacts.

    Args:
        folder_path: Folder expected to contain the persisted vector-store
            files.

    Raises:
        FileNotFoundError: If the folder is missing one or more required
            persisted artifacts.
    """
    if _has_persisted_vector_store(folder_path):
        return

    required_artifacts = ", ".join((METADATA_FILE_NAME, VECTORS_FILE_NAME))
    raise FileNotFoundError(
        f"No persisted vector store found in {folder_path}. "
        f"Required persisted artifacts: {required_artifacts}."
    )


def read_index_source_file(*, folder_path: str) -> str | None:
    """Read the recorded source-file path from persisted metadata.

    Args:
        folder_path: Folder that contains the persisted vector-store artifacts.

    Returns:
        The source-file path recorded in metadata, or ``None`` when the value
        is absent.
    """
    metadata = _read_metadata(folder_path)
    return metadata.get(INDEX_SOURCE_FILE_KEY)


def read_embedding_model_name(*, folder_path: str) -> str | None:
    """Read the recorded embedding model name from persisted metadata.

    Args:
        folder_path: Folder that contains the persisted vector-store artifacts.

    Returns:
        The embedding model name recorded in metadata, or ``None`` when the
        value is absent.
    """
    metadata = _read_metadata(folder_path)
    return metadata.get(EMBEDDING_MODEL_NAME_KEY)


def _has_persisted_vector_store(folder_path: str) -> bool:
    """Return whether the expected persisted ClassifAI files are present."""
    metadata_path = _metadata_path(folder_path)
    vectors_path = _vectors_path(folder_path)
    return (
        os.path.isdir(folder_path)
        and os.path.exists(metadata_path)
        and os.path.exists(vectors_path)
    )


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


def _metadata_path(folder_path: str) -> str:
    """Return the metadata file path for a persisted store folder."""
    return os.path.join(folder_path, METADATA_FILE_NAME)


def _vectors_path(folder_path: str) -> str:
    """Return the vectors file path for a persisted store folder."""
    return os.path.join(folder_path, VECTORS_FILE_NAME)
