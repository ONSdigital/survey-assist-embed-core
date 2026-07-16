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
    VectoriserClass,
    _ClassifaiVectorIndex,
    _resolve_local_path,
    resolve_model_name,
)
from survey_assist_embed_core.adapters.storage import DownloadedVectorStore

EXPECTED_LOADED_VECTOR_COUNT = 42
EXPECTED_BUILT_VECTOR_COUNT = 7
EXPECTED_SEARCH_LIMIT = 5
EXPECTED_SEARCH_SCORE = 0.9
EXPECTED_BATCH_SIZE = 32


def make_search_output(rows: list[dict[str, object]]) -> VectorStoreSearchOutput:
    """Build a valid ClassifAI search output for tests."""
    return VectorStoreSearchOutput.from_data(
        {
            "query_id": [str(row.get("query_id", "q1")) for row in rows],
            "query_text": [str(row.get("query_text", "test query")) for row in rows],
            "doc_label": [str(row["doc_label"]) for row in rows],
            "doc_text": [str(row["doc_text"]) for row in rows],
            "rank": [int(row.get("rank", index + 1)) for index, row in enumerate(rows)],
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
    assert backend.config.settings == {
        "embedding_model_name": "sentence-transformers/persisted-model",
        "vectoriser_class": "onnx",
    }
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
        quiet_mode=True,
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
            "build_vectoriser",
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

    mock_build_vectoriser.assert_called_once_with(
        "sentence-transformers/other",
        vectoriser_class=VectoriserClass.ONNX,
    )
    mock_vector_store.assert_called_once_with(
        file_name="source.csv",
        data_type="csv",
        vectoriser=vectoriser,
        batch_size=128,
        meta_data=None,
        output_dir="vector_store",
        overwrite=True,
        hooks=None,
        quiet_mode=True,
    )
    mock_write_vector_store_metadata.assert_called_once_with(
        folder_path="vector_store",
        index_source_file="source.csv",
        embedding_model_name="sentence-transformers/other",
        vectoriser_class="onnx",
    )


def test_build_classifai_vector_store_artifacts_allows_batch_size_override() -> None:
    vectoriser = object()

    with (
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "build_vectoriser",
            return_value=vectoriser,
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend.VectorStore",
        ) as mock_vector_store,
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "write_vector_store_metadata",
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "_resolve_local_path",
            side_effect=contextmanager(lambda path: iter([path])),
        ),
    ):
        build_classifai_vector_store_artifacts(
            index_source_file="source.csv",
            output_dir="vector_store",
            batch_size=32,
        )

    assert mock_vector_store.call_args.kwargs["batch_size"] == EXPECTED_BATCH_SIZE


def test_classifai_resolve_local_path_yields_path_unchanged(tmp_path) -> None:
    local_file = str(tmp_path / "source.csv")
    with _resolve_local_path(local_file) as resolved:
        assert resolved == local_file


def test_classifai_normalise_model_name_prepends_prefix() -> None:
    # bare name gets the default org prepended
    assert (
        resolve_model_name("all-MiniLM-L6-v2")
        == "sentence-transformers/all-MiniLM-L6-v2"
    )
    # fully qualified names of any org pass through unchanged
    assert (
        resolve_model_name("sentence-transformers/all-MiniLM-L6-v2")
        == "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert resolve_model_name("BAAI/bge-small-en-v1.5") == "BAAI/bge-small-en-v1.5"
    assert resolve_model_name("intfloat/e5-small-v2") == "intfloat/e5-small-v2"


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("ONNX", VectoriserClass.ONNX),
        ("onnx", VectoriserClass.ONNX),
        ("onnx_vectoriser", VectoriserClass.ONNX),
        ("onnx_vectorizer", VectoriserClass.ONNX),
        ("OnnxVectoriser", VectoriserClass.ONNX),
        ("HF", VectoriserClass.HUGGINGFACE),
        ("hf", VectoriserClass.HUGGINGFACE),
        ("huggingface", VectoriserClass.HUGGINGFACE),
        ("normalised_hf_vectoriser", VectoriserClass.HUGGINGFACE),
        ("normalized_hf_vectorizer", VectoriserClass.HUGGINGFACE),
    ],
)
def test_resolve_vectoriser_class_accepts_aliases(
    raw_value: str,
    expected: VectoriserClass,
):
    """Accept mixed-case and common alias forms for vectoriser class selection."""
    assert VectoriserClass.resolve(raw_value) == expected


def test_resolve_vectoriser_class_rejects_unknown_alias() -> None:
    """Reject unsupported class names after alias normalisation."""
    with pytest.raises(
        ValueError,
        match="must resolve to either 'onnx' or 'huggingface'",
    ):
        VectoriserClass.resolve("bert")


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
            "build_vectoriser",
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

    called_input = fake_store.search.call_args.args[0]
    assert called_input.id.to_list() == ["q1"]
    assert called_input.query.to_list() == ["dog"]
    assert fake_store.search.call_args.kwargs["n_results"] == EXPECTED_SEARCH_LIMIT
    assert results[0]["doc_label"] == "02"
    assert results[0]["doc_text"] == "dog"
    assert results[0]["score"] == EXPECTED_SEARCH_SCORE


def test_classifai_vector_backend_search_many_batches_queries(tmp_path) -> None:
    rows = [
        {
            "query_id": "q2",
            "query_text": "cat",
            "doc_text": "cat",
            "score": 0.8,
            "doc_label": "01",
            "rank": 1,
        },
        {
            "query_id": "q1",
            "query_text": "dog",
            "doc_text": "dog",
            "score": EXPECTED_SEARCH_SCORE,
            "doc_label": "02",
            "rank": 1,
        },
    ]
    folder_path = str(tmp_path / "vector_store")
    fake_store = SimpleNamespace(
        num_vectors=2,
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

    results = index.search_many(["dog", "cat"], limit=EXPECTED_SEARCH_LIMIT)

    called_input = fake_store.search.call_args.args[0]
    assert called_input.id.to_list() == ["q1", "q2"]
    assert called_input.query.to_list() == ["dog", "cat"]
    assert fake_store.search.call_args.kwargs["n_results"] == EXPECTED_SEARCH_LIMIT
    assert results == [
        [
            {
                "query_id": "q1",
                "query_text": "dog",
                "doc_label": "02",
                "doc_text": "dog",
                "rank": 1,
                "score": EXPECTED_SEARCH_SCORE,
            }
        ],
        [
            {
                "query_id": "q2",
                "query_text": "cat",
                "doc_label": "01",
                "doc_text": "cat",
                "rank": 1,
                "score": 0.8,
            }
        ],
    ]


def test_classifai_vector_backend_search_many_returns_empty_for_no_queries() -> None:
    fake_store = SimpleNamespace(num_vectors=2, search=MagicMock())
    index = _ClassifaiVectorIndex(fake_store)

    assert index.search_many([], limit=EXPECTED_SEARCH_LIMIT) == []
    fake_store.search.assert_not_called()


def test_classifai_vector_backend_config_reports_loaded_model_name() -> None:
    backend = ClassifaiVectorBackend()
    backend._set_embedding_model_name("other")

    assert backend.config.backend_name == "classifai"
    assert backend.config.settings == {
        "embedding_model_name": "sentence-transformers/other",
        "vectoriser_class": None,
    }


def test_classifai_vector_backend_build_vectoriser_memoizes_instance() -> None:
    backend = ClassifaiVectorBackend()
    backend._set_embedding_model_name("other")
    fake_vectoriser = object()

    with patch(
        "survey_assist_embed_core.adapters.classifai.vector_backend.build_vectoriser",
        return_value=fake_vectoriser,
    ) as mock_vectoriser:
        first = backend._get_vectoriser()
        second = backend._get_vectoriser()

    assert first is fake_vectoriser
    assert second is fake_vectoriser
    mock_vectoriser.assert_called_once_with(
        embedding_model_name="sentence-transformers/other",
        vectoriser_class=None,
    )


def test_build_classifai_vector_store_artifacts_passes_explicit_vectoriser_class() -> (
    None
):
    with (
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend.build_vectoriser",
            return_value=object(),
        ) as mock_build_vectoriser,
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend.VectorStore",
            return_value=SimpleNamespace(num_vectors=1, search=MagicMock()),
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai"
            + ".vector_backend.write_vector_store_metadata",
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend._resolve_local_path",
            side_effect=contextmanager(lambda path: iter([path])),
        ),
    ):
        build_classifai_vector_store_artifacts(
            index_source_file="source.csv",
            output_dir="vector_store",
            embedding_model_name="other",
            vectoriser_class="HF",
        )

    mock_build_vectoriser.assert_called_once_with(
        "sentence-transformers/other",
        vectoriser_class=VectoriserClass.HUGGINGFACE,
    )


def test_classifai_vector_backend_build_vectoriser_uses_configured_kind() -> None:
    backend = ClassifaiVectorBackend()
    backend._set_embedding_model_name("other")
    backend._set_vectoriser_class("HF")

    with patch(
        "survey_assist_embed_core.adapters.classifai.vector_backend.build_vectoriser",
        return_value=object(),
    ) as mock_vectoriser:
        backend._get_vectoriser()

    mock_vectoriser.assert_called_once_with(
        embedding_model_name="sentence-transformers/other",
        vectoriser_class=VectoriserClass.HUGGINGFACE,
    )


def test_classifai_vector_backend_load_uses_runtime_vectoriser_class(
    tmp_path,
) -> None:
    backend = ClassifaiVectorBackend()
    vectoriser = object()
    folder_path = str(tmp_path / "vector_store")
    fake_store = SimpleNamespace(num_vectors=1, search=MagicMock())

    with (
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "ensure_persisted_vector_store",
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "read_embedding_model_name",
            return_value="persisted-model",
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "build_vectoriser",
            return_value=vectoriser,
        ) as mock_build_vectoriser,
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "VectorStore.from_filespace",
            return_value=fake_store,
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "read_index_source_file",
            return_value=None,
        ),
    ):
        backend.load(folder_path=folder_path, vectoriser_class="HF")

    mock_build_vectoriser.assert_called_once_with(
        embedding_model_name="sentence-transformers/persisted-model",
        vectoriser_class=VectoriserClass.HUGGINGFACE,
    )


def test_classifai_vector_backend_load_warns_on_vectoriser_class_conflict(
    tmp_path,
) -> None:
    backend = ClassifaiVectorBackend()
    vectoriser = object()
    folder_path = str(tmp_path / "vector_store")
    fake_store = SimpleNamespace(num_vectors=1, search=MagicMock())

    with (
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "ensure_persisted_vector_store",
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "read_embedding_model_name",
            return_value="persisted-model",
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "read_vectoriser_class",
            return_value="onnx",
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "build_vectoriser",
            return_value=vectoriser,
        ) as mock_build_vectoriser,
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "VectorStore.from_filespace",
            return_value=fake_store,
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "read_index_source_file",
            return_value=None,
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend.logger.warning",
        ) as mock_warning,
    ):
        backend.load(folder_path=folder_path, vectoriser_class="HF")

    mock_build_vectoriser.assert_called_once_with(
        embedding_model_name="sentence-transformers/persisted-model",
        vectoriser_class=VectoriserClass.HUGGINGFACE,
    )
    mock_warning.assert_called_once_with(
        "Vectoriser class provided does not match persisted metadata."
        "Using provided class.",
        provided_class="huggingface",
        persisted_class="onnx",
    )


def test_classifai_vector_backend_set_embedding_model_name_noops_when_unchanged() -> (
    None
):
    backend = ClassifaiVectorBackend()
    fake_vectoriser = object()
    backend._embedding_model_name = "sentence-transformers/other"
    backend._vectoriser = fake_vectoriser

    backend._set_embedding_model_name("other")

    assert backend._embedding_model_name == "sentence-transformers/other"
    assert backend._vectoriser is fake_vectoriser


def test_classifai_vector_backend_load_requires_embedding_model_metadata(
    tmp_path,
) -> None:
    backend = ClassifaiVectorBackend()
    folder_path = str(tmp_path / "vector_store")
    fake_store = SimpleNamespace(num_vectors=1, search=MagicMock())

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
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "VectorStore.from_filespace",
            return_value=fake_store,
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "read_index_source_file",
            return_value=None,
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend.logger.warning",
        ) as mock_warning,
    ):
        index, index_source_file = backend.load(folder_path=folder_path)

    assert index.num_vectors == 1
    assert index_source_file is None
    assert backend.config.settings == {
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "vectoriser_class": "onnx",
    }
    mock_warning.assert_called_once_with(
        "No embedding model metadata found in persisted vector store. Using default model.",
        resolved_model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
