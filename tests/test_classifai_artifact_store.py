# pylint: disable=missing-function-docstring
"""Tests for the ClassifAI artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from survey_assist_embed_core.adapters.classifai import artifacts


def test_classifai_artifact_store_detects_persisted_store(tmp_path: Path) -> None:
    folder_path = tmp_path / "vector_store"
    folder_path.mkdir()
    (folder_path / "metadata.json").write_text("{}", encoding="utf-8")
    (folder_path / "vectors.parquet").write_text("dummy", encoding="utf-8")

    artifacts.ensure_persisted_vector_store(folder_path=str(folder_path))


def test_classifai_artifact_store_error_uses_default_layout_names(
    tmp_path: Path,
) -> None:
    folder_path = tmp_path / "vector_store"
    folder_path.mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        artifacts.ensure_persisted_vector_store(folder_path=str(folder_path))

    assert "metadata.json, vectors.parquet" in str(exc_info.value)


def test_classifai_artifact_store_reads_and_writes_source_file(
    tmp_path: Path,
) -> None:
    folder_path = tmp_path / "vector_store"
    folder_path.mkdir()

    artifacts.write_index_source_file(
        folder_path=str(folder_path),
        index_source_file="source.csv",
    )

    metadata = json.loads((folder_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["index_source_file"] == "source.csv"
    assert (
        artifacts.read_index_source_file(folder_path=str(folder_path)) == "source.csv"
    )


def test_classifai_artifact_store_reads_and_writes_embedding_model_name(
    tmp_path: Path,
) -> None:
    folder_path = tmp_path / "vector_store"
    folder_path.mkdir()

    artifacts.write_embedding_model_name(
        folder_path=str(folder_path),
        embedding_model_name="all-MiniLM-L6-v2",
    )

    metadata = json.loads((folder_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["embedding_model_name"] == "all-MiniLM-L6-v2"
    assert (
        artifacts.read_embedding_model_name(folder_path=str(folder_path))
        == "all-MiniLM-L6-v2"
    )


def test_classifai_artifact_store_writes_metadata_in_single_operation(
    tmp_path: Path,
) -> None:
    folder_path = tmp_path / "vector_store"
    folder_path.mkdir()

    artifacts.write_vector_store_metadata(
        folder_path=str(folder_path),
        index_source_file="source.csv",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    metadata = json.loads((folder_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata == {
        "index_source_file": "source.csv",
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    }
