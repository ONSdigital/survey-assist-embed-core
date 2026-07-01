# pylint: disable=missing-function-docstring, protected-access
"""Tests for the ClassifAI vector backend adapter."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from classifai.indexers import VectorStoreSearchOutput

from survey_assist_embed_core.adapters.classifai import (
    ClassifaiVectorBackend,
    build_classifai_vector_store_artifacts,
)
from survey_assist_embed_core.adapters.classifai.vector_backend import (
    _resolve_local_path,
)
from survey_assist_embed_core.adapters.storage import DownloadedVectorStore

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
    backend = ClassifaiVectorBackend()
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
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "ensure_persisted_vector_store",
        ) as mock_ensure_store,
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "read_embedding_model_name",
            return_value="persisted-model",
        ) as mock_read_embedding_model_name,
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "read_index_source_file",
            return_value="source.csv",
        ) as mock_read_index_source_file,
    ):
        index, index_source_file = backend.load(folder_path=folder_path)

    assert index.num_vectors == EXPECTED_LOADED_VECTOR_COUNT
    assert index_source_file == "source.csv"
    assert backend.config.settings == {"embedding_model_name": "persisted-model"}
    mock_build_vectoriser.assert_called_once_with()
    mock_ensure_store.assert_called_once_with(
        folder_path=folder_path,
    )
    mock_read_embedding_model_name.assert_called_once_with(
        folder_path=folder_path,
    )
    mock_read_index_source_file.assert_called_once_with(
        folder_path=folder_path,
    )
    mock_from_filespace.assert_called_once_with(
        folder_path=folder_path,
        vectoriser=vectoriser,
        hooks=None,
    )


def test_build_classifai_vector_store_artifacts_uses_expected_args() -> None:
    vectoriser = object()
    fake_store = SimpleNamespace(
        num_vectors=EXPECTED_BUILT_VECTOR_COUNT,
        search=MagicMock(),
    )

    with (
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "NormalisedHFVectoriser",
            return_value=vectoriser,
        ) as mock_build_vectoriser,
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend.VectorStore",
            return_value=fake_store,
        ) as mock_vector_store,
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "write_vector_store_metadata",
        ) as mock_write_vector_store_metadata,
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "_resolve_local_path",
            side_effect=contextmanager(lambda path: iter([path])),
        ),
    ):
        build_classifai_vector_store_artifacts(
            index_source_file="source.csv",
            output_dir="vector_store",
            embedding_model_name="other",
        )

    mock_build_vectoriser.assert_called_once_with(model_name="other")
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
    mock_write_vector_store_metadata.assert_called_once_with(
        folder_path="vector_store",
        index_source_file="source.csv",
        embedding_model_name="other",
    )


def test_classifai_resolve_local_path_yields_path_unchanged(tmp_path) -> None:
    local_file = str(tmp_path / "source.csv")
    with _resolve_local_path(local_file) as resolved:
        assert resolved == local_file


def test_build_classifai_vector_store_artifacts_downloads_gcs_source_file(
    tmp_path,
) -> None:
    vectoriser = object()
    fake_store = SimpleNamespace(
        num_vectors=EXPECTED_BUILT_VECTOR_COUNT,
        search=MagicMock(),
    )
    downloaded_path = str(tmp_path / "downloaded.csv")
    downloaded = DownloadedVectorStore(
        path=downloaded_path,
        temp_dir=SimpleNamespace(name=str(tmp_path), cleanup=lambda: None),
    )

    with (
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "NormalisedHFVectoriser",
            return_value=vectoriser,
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend.VectorStore",
            return_value=fake_store,
        ) as mock_vector_store,
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "write_vector_store_metadata",
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend.is_gcs_path",
            return_value=True,
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "download_one_file_from_gcs",
            return_value=downloaded,
        ) as mock_download,
    ):
        build_classifai_vector_store_artifacts(
            index_source_file="gs://my-bucket/source.csv",
            output_dir="vector_store",
            embedding_model_name="other",
        )

    mock_download.assert_called_once_with("gs://my-bucket/source.csv")
    assert mock_vector_store.call_args.kwargs["file_name"] == downloaded_path


def test_classifai_vector_backend_search_returns_records(tmp_path) -> None:
    rows = [{"doc_text": "dog", "score": EXPECTED_SEARCH_SCORE, "doc_label": "02"}]
    folder_path = str(tmp_path / "vector_store")
    fake_store = SimpleNamespace(
        num_vectors=1,
        search=MagicMock(return_value=make_search_output(rows)),
    )
    backend = ClassifaiVectorBackend()

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
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "ensure_persisted_vector_store",
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "read_embedding_model_name",
            return_value="other",
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "read_index_source_file",
            return_value=None,
        ),
    ):
        index, _ = backend.load(folder_path=folder_path)

    results = index.search("dog", limit=EXPECTED_SEARCH_LIMIT)

    assert fake_store.search.call_args.kwargs["n_results"] == EXPECTED_SEARCH_LIMIT
    assert results[0]["doc_label"] == "02"
    assert results[0]["doc_text"] == "dog"
    assert results[0]["score"] == EXPECTED_SEARCH_SCORE


def test_classifai_vector_backend_config_reports_loaded_model_name() -> None:
    backend = ClassifaiVectorBackend()
    backend._set_embedding_model_name("other")

    assert backend.config.backend_name == "classifai"
    assert backend.config.settings == {"embedding_model_name": "other"}


def test_classifai_vector_backend_build_vectoriser_memoizes_instance() -> None:
    backend = ClassifaiVectorBackend()
    backend._set_embedding_model_name("other")
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
    mock_vectoriser.assert_called_once_with(model_name="other")


def test_classifai_vector_backend_set_embedding_model_name_noops_when_unchanged() -> (
    None
):
    backend = ClassifaiVectorBackend()
    fake_vectoriser = object()
    backend._embedding_model_name = "other"
    backend._vectoriser = fake_vectoriser

    backend._set_embedding_model_name("other")

    assert backend._embedding_model_name == "other"
    assert backend._vectoriser is fake_vectoriser


def test_classifai_vector_backend_get_vectoriser_requires_loaded_model_name() -> None:
    backend = ClassifaiVectorBackend()

    with pytest.raises(
        ValueError,
        match="embedding_model_name must be loaded from persisted metadata",
    ):
        backend._get_vectoriser()


def test_classifai_vector_backend_load_requires_embedding_model_metadata(
    tmp_path,
) -> None:
    backend = ClassifaiVectorBackend()
    folder_path = str(tmp_path / "vector_store")

    with (
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "ensure_persisted_vector_store",
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "read_embedding_model_name",
            return_value=None,
        ),
        pytest.raises(
            FileNotFoundError,
            match="No embedding model metadata found in persisted vector store",
        ),
    ):
        backend.load(folder_path=folder_path)
