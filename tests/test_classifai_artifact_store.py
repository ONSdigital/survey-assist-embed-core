# pylint: disable=missing-function-docstring
"""Tests for the ClassifAI artifact-store adapter."""

from __future__ import annotations

import json
from pathlib import Path

from survey_assist_embed_core.adapters.classifai import ClassifaiArtifactStore


def test_classifai_artifact_store_detects_persisted_store(tmp_path: Path) -> None:
    artifact_store = ClassifaiArtifactStore()
    folder_path = tmp_path / "vector_store"
    folder_path.mkdir()
    (folder_path / "metadata.json").write_text("{}", encoding="utf-8")
    (folder_path / "vectors.parquet").write_text("dummy", encoding="utf-8")

    artifact_store.ensure_persisted_vector_store(folder_path=str(folder_path))


def test_classifai_artifact_store_error_uses_configured_layout_names(
    tmp_path: Path,
) -> None:
    artifact_store = ClassifaiArtifactStore(
        metadata_file_name="store-metadata.json",
        vectors_file_name="store-vectors.parquet",
    )
    folder_path = tmp_path / "vector_store"
    folder_path.mkdir()

    try:
        artifact_store.ensure_persisted_vector_store(folder_path=str(folder_path))
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")

    assert "store-metadata.json, store-vectors.parquet" in message


def test_classifai_artifact_store_reads_and_writes_source_file(
    tmp_path: Path,
) -> None:
    artifact_store = ClassifaiArtifactStore()
    folder_path = tmp_path / "vector_store"
    folder_path.mkdir()

    artifact_store.write_index_source_file(
        folder_path=str(folder_path),
        index_source_file="source.csv",
    )

    metadata = json.loads((folder_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["index_source_file"] == "source.csv"
    assert (
        artifact_store.read_index_source_file(folder_path=str(folder_path))
        == "source.csv"
    )


def test_classifai_artifact_store_detects_vectors_file(tmp_path: Path) -> None:
    artifact_store = ClassifaiArtifactStore()
    folder_path = tmp_path / "vector_store"
    folder_path.mkdir()
    (folder_path / "vectors.parquet").write_text("dummy", encoding="utf-8")

    assert artifact_store.has_persisted_vectors_file(folder_path=str(folder_path))


def test_classifai_artifact_store_uses_configured_layout_names(
    tmp_path: Path,
) -> None:
    artifact_store = ClassifaiArtifactStore(
        metadata_file_name="store-metadata.json",
        vectors_file_name="store-vectors.parquet",
        index_source_file_key="source_path",
    )
    folder_path = tmp_path / "vector_store"
    folder_path.mkdir()
    (folder_path / "store-vectors.parquet").write_text("dummy", encoding="utf-8")

    artifact_store.write_index_source_file(
        folder_path=str(folder_path),
        index_source_file="source.csv",
    )

    metadata = json.loads(
        (folder_path / "store-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["source_path"] == "source.csv"
    artifact_store.ensure_persisted_vector_store(folder_path=str(folder_path))
    assert (
        artifact_store.read_index_source_file(folder_path=str(folder_path))
        == "source.csv"
    )
