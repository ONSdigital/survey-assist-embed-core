# pylint: disable=missing-function-docstring, protected-access
"""Tests for the ClassifAI vector backend adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from classifai.indexers import VectorStoreSearchOutput

from survey_assist_embed_core.adapters.classifai import (
    ClassifaiArtifactStore,
    ClassifaiVectorBackend,
)

EXPECTED_LOADED_VECTOR_COUNT = 42
EXPECTED_BUILT_VECTOR_COUNT = 7
EXPECTED_SEARCH_LIMIT = 5
EXPECTED_SEARCH_SCORE = 0.9


def make_search_output(rows: list[dict[str, object]]) -> VectorStoreSearchOutput:
    """Build a valid ClassifAI search output for tests."""
    return VectorStoreSearchOutput.from_data(
        {
            "query_id": ["q1"] * len(rows),
            "query_text": ["test query"] * len(rows),
            "doc_label": [str(row["doc_label"]) for row in rows],
            "doc_text": [str(row["doc_text"]) for row in rows],
            "rank": list(range(1, len(rows) + 1)),
            "score": [float(cast(float | int, row["score"])) for row in rows],
        }
    )


def test_classifai_vector_backend_load_uses_from_filespace(tmp_path) -> None:
    artifact_store = SimpleNamespace(
        ensure_persisted_vector_store=MagicMock(),
        read_index_source_file=MagicMock(return_value="source.csv"),
        has_persisted_vectors_file=MagicMock(return_value=True),
        write_index_source_file=MagicMock(),
    )
    backend = ClassifaiVectorBackend(
        embedding_model_name="other",
        artifact_store=artifact_store,
    )
    vectoriser = object()
    folder_path = str(tmp_path / "vector_store")
    fake_store = SimpleNamespace(
        num_vectors=EXPECTED_LOADED_VECTOR_COUNT,
        search=MagicMock(),
    )

    with (
        patch.object(
            backend,
            "_get_vectoriser",
            return_value=vectoriser,
        ) as mock_build_vectoriser,
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "VectorStore.from_filespace",
            return_value=fake_store,
        ) as mock_from_filespace,
    ):
        index, index_source_file = backend.load(folder_path=folder_path)

    assert index.num_vectors == EXPECTED_LOADED_VECTOR_COUNT
    assert index_source_file == "source.csv"
    mock_build_vectoriser.assert_called_once_with()
    artifact_store.ensure_persisted_vector_store.assert_called_once_with(
        folder_path=folder_path,
    )
    artifact_store.read_index_source_file.assert_called_once_with(
        folder_path=folder_path,
    )
    mock_from_filespace.assert_called_once_with(
        folder_path=folder_path,
        vectoriser=vectoriser,
        hooks=None,
    )


def test_classifai_vector_backend_build_uses_expected_args() -> None:
    artifact_store = SimpleNamespace(
        ensure_persisted_vector_store=MagicMock(),
        read_index_source_file=MagicMock(return_value="source.csv"),
        has_persisted_vectors_file=MagicMock(return_value=False),
        write_index_source_file=MagicMock(),
    )
    backend = ClassifaiVectorBackend(
        embedding_model_name="other",
        artifact_store=artifact_store,
    )
    vectoriser = object()
    fake_store = SimpleNamespace(
        num_vectors=EXPECTED_BUILT_VECTOR_COUNT,
        search=MagicMock(),
    )

    with (
        patch.object(
            backend,
            "_get_vectoriser",
            return_value=vectoriser,
        ) as mock_build_vectoriser,
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend.VectorStore",
            return_value=fake_store,
        ) as mock_vector_store,
    ):
        backend.build(
            file_name="source.csv",
            output_dir="vector_store",
            index_source_file="source.csv",
        )

    mock_build_vectoriser.assert_called_once_with()
    mock_vector_store.assert_called_once_with(
        file_name="source.csv",
        data_type="csv",
        vectoriser=vectoriser,
        batch_size=8,
        meta_data=None,
        output_dir="vector_store",
        overwrite=True,
        hooks=None,
    )
    artifact_store.write_index_source_file.assert_called_once_with(
        folder_path="vector_store",
        index_source_file="source.csv",
    )


def test_classifai_vector_backend_search_returns_records(tmp_path) -> None:
    rows = [{"doc_text": "dog", "score": EXPECTED_SEARCH_SCORE, "doc_label": "02"}]
    folder_path = str(tmp_path / "vector_store")
    fake_store = SimpleNamespace(
        num_vectors=1,
        search=MagicMock(return_value=make_search_output(rows)),
    )
    artifact_store = SimpleNamespace(
        ensure_persisted_vector_store=MagicMock(),
        read_index_source_file=MagicMock(return_value=None),
        has_persisted_vectors_file=MagicMock(return_value=True),
        write_index_source_file=MagicMock(),
    )
    backend = ClassifaiVectorBackend(
        embedding_model_name="other",
        artifact_store=artifact_store,
    )

    with (
        patch.object(
            backend,
            "_get_vectoriser",
            return_value=object(),
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "VectorStore.from_filespace",
            return_value=fake_store,
        ),
    ):
        index, _ = backend.load(folder_path=folder_path)

    results = index.search("dog", limit=EXPECTED_SEARCH_LIMIT)

    assert fake_store.search.call_args.kwargs["n_results"] == EXPECTED_SEARCH_LIMIT
    assert results[0]["doc_label"] == "02"
    assert results[0]["doc_text"] == "dog"
    assert results[0]["score"] == EXPECTED_SEARCH_SCORE


def test_classifai_vector_backend_config_reports_model_name() -> None:
    backend = ClassifaiVectorBackend(embedding_model_name="other")

    assert backend.config.backend_name == "classifai"
    assert backend.config.settings == {"embedding_model_name": "other"}


def test_classifai_vector_backend_build_vectoriser_memoizes_instance() -> None:
    backend = ClassifaiVectorBackend(embedding_model_name="other")
    fake_vectoriser = object()

    with patch(
        "survey_assist_embed_core.adapters.classifai.vector_backend."
        "NormalisedHFVectoriser",
        return_value=fake_vectoriser,
    ) as mock_vectoriser:
        first = backend._get_vectoriser()
        second = backend._get_vectoriser()

    assert first is fake_vectoriser
    assert second is fake_vectoriser
    mock_vectoriser.assert_called_once_with(model_name="sentence-transformers/other")


def test_classifai_vector_backend_has_persisted_store_uses_artifact_store() -> None:
    artifact_store = SimpleNamespace(
        ensure_persisted_vector_store=MagicMock(),
        read_index_source_file=MagicMock(return_value=None),
        has_persisted_vectors_file=MagicMock(return_value=True),
        write_index_source_file=MagicMock(),
    )
    backend = ClassifaiVectorBackend(artifact_store=artifact_store)

    assert backend.has_persisted_store(folder_path="vector_store") is True
    artifact_store.has_persisted_vectors_file.assert_called_once_with(
        folder_path="vector_store",
    )


def test_classifai_vector_backend_load_uses_custom_artifact_layout_names(
    tmp_path,
) -> None:
    class _CustomClassifaiArtifactStore(ClassifaiArtifactStore):
        METADATA_FILE_NAME = "store-metadata.json"
        VECTORS_FILE_NAME = "store-vectors.parquet"

    backend = ClassifaiVectorBackend(
        artifact_store=_CustomClassifaiArtifactStore(),
    )
    folder_path = tmp_path / "vector_store"
    folder_path.mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        backend.load(folder_path=str(folder_path))

    assert "store-metadata.json, store-vectors.parquet" in str(exc_info.value)
