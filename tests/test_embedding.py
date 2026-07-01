# pylint: disable=missing-function-docstring, protected-access, redefined-outer-name
"""Tests for the shared retrieval embedding handler and vectoriser."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from survey_assist_embed_core.adapters.classifai import (
    ClassifaiVectorBackend,
    NormalisedHFVectoriser,
)
from survey_assist_embed_core.adapters.storage.gcs import DownloadedVectorStore
from survey_assist_embed_core.embed import EmbeddingHandler
from survey_assist_embed_core.models import SearchIndexResponse, VectorBackendConfig

EXPECTED_TOY_INDEX_SIZE = 4
EXPECTED_MULTI_RESULTS = 4
EXPECTED_TOP_DISTANCE = 0.1
EXPECTED_CONFIG_K_MATCHES = 5
EXPECTED_CONFIG_INDEX_SIZE = 7


@pytest.fixture
def embedding_handler_for_embed(tmp_path: Path) -> EmbeddingHandler:
    placeholder_store = SimpleNamespace(num_vectors=1)

    with patch(
        "survey_assist_embed_core.embed.embedding."
        "EmbeddingHandler._load_existing_vector_store",
        return_value=(placeholder_store, "mock-source.csv"),
    ):
        handler = EmbeddingHandler(db_dir=str(tmp_path / "vector_store"))

    return handler


@pytest.fixture
def embedding_handler_search(tmp_path: Path) -> EmbeddingHandler:
    rows = [
        {
            "doc_text": "dog",
            "score": 0.99,
            "doc_label": "02",
        },
        {
            "doc_text": "cat",
            "score": 0.75,
            "doc_label": "01",
        },
    ]
    fake_store = SimpleNamespace(
        num_vectors=EXPECTED_TOY_INDEX_SIZE,
        search=MagicMock(return_value=rows),
    )
    with patch(
        "survey_assist_embed_core.embed.embedding."
        "EmbeddingHandler._load_existing_vector_store",
        return_value=(fake_store, "mock-source.csv"),
    ):
        handler = EmbeddingHandler(db_dir=str(tmp_path / "vector_store"))

    return handler


def test_embedding_handler_init_sets_vector_store(tmp_path: Path) -> None:
    built_store = SimpleNamespace(num_vectors=EXPECTED_TOY_INDEX_SIZE)

    with patch(
        "survey_assist_embed_core.embed.embedding."
        "EmbeddingHandler._load_existing_vector_store",
        return_value=(built_store, "mock-source.csv"),
    ):
        handler = EmbeddingHandler(db_dir=str(tmp_path / "vector_store"))

    assert handler.vector_store is built_store


def test_embedding_handler_init_sets_index_size(tmp_path: Path) -> None:
    built_store = SimpleNamespace(num_vectors=EXPECTED_TOY_INDEX_SIZE)

    with patch(
        "survey_assist_embed_core.embed.embedding."
        "EmbeddingHandler._load_existing_vector_store",
        return_value=(built_store, "mock-source.csv"),
    ):
        handler = EmbeddingHandler(db_dir=str(tmp_path / "vector_store"))

    assert handler.index_size == EXPECTED_TOY_INDEX_SIZE


def test_embedding_handler_init_rejects_empty_vector_store(tmp_path: Path) -> None:
    empty_store = SimpleNamespace(num_vectors=0)

    with (
        patch(
            "survey_assist_embed_core.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(empty_store, "mock-source.csv"),
        ),
        pytest.raises(
            ValueError,
            match="Persisted vector store contains no vectors",
        ),
    ):
        EmbeddingHandler(db_dir=str(tmp_path / "vector_store"))


def test_embedding_handler_init_keeps_falsey_explicit_dependencies(
    tmp_path: Path,
) -> None:
    class _FalseyDependency:  # pylint: disable=too-few-public-methods
        def __bool__(self) -> bool:
            return False

        @property
        def config(self) -> VectorBackendConfig:
            return VectorBackendConfig(backend_name="test-backend", settings={})

    built_store = SimpleNamespace(num_vectors=EXPECTED_TOY_INDEX_SIZE)
    backend = _FalseyDependency()

    with patch(
        "survey_assist_embed_core.embed.embedding."
        "EmbeddingHandler._load_existing_vector_store",
        return_value=(built_store, "mock-source.csv"),
    ):
        handler = EmbeddingHandler(
            db_dir=str(tmp_path / "vector_store"),
            backend=backend,
        )

    assert handler._backend is backend


def test_search_index(embedding_handler_search: EmbeddingHandler) -> None:
    response = embedding_handler_search.search_index("mens best friend")

    assert isinstance(response, SearchIndexResponse)
    assert response.results[0].code == "02"
    assert response.results[0].title == "dog"
    assert response.results[0].distance == pytest.approx(0.01)


def test_search_index_uses_backend_search_rows(tmp_path: Path) -> None:
    rows = [{"doc_text": "dog", "score": 0.9, "doc_label": "02"}]
    fake_store = SimpleNamespace(
        num_vectors=1,
        search=MagicMock(return_value=rows),
    )

    with patch(
        "survey_assist_embed_core.embed.embedding."
        "EmbeddingHandler._load_existing_vector_store",
        return_value=(fake_store, "mock-source.csv"),
    ):
        handler = EmbeddingHandler(db_dir=str(tmp_path / "vector_store"))

    results = handler.search_index("dog")

    assert isinstance(results, SearchIndexResponse)
    assert results.results[0].code == "02"
    assert results.results[0].title == "dog"


def test_search_index_multi(tmp_path: Path) -> None:
    fake_store = SimpleNamespace(
        num_vectors=EXPECTED_TOY_INDEX_SIZE,
        search_many=MagicMock(
            return_value=[
                [
                    {"doc_text": "fish", "score": 0.6, "doc_label": "03"},
                    {"doc_text": "lizard", "score": 0.4, "doc_label": "04"},
                ],
                [
                    {"doc_text": "fish", "score": 0.9, "doc_label": "03"},
                    {"doc_text": "lizard", "score": 0.8, "doc_label": "04"},
                ],
            ]
        ),
    )

    with patch(
        "survey_assist_embed_core.embed.embedding."
        "EmbeddingHandler._load_existing_vector_store",
        return_value=(fake_store, "mock-source.csv"),
    ):
        handler = EmbeddingHandler(db_dir=str(tmp_path / "vector_store"))

    with patch.object(handler, "spell", side_effect=lambda text: text):
        response = handler.search_index_multi(["has gills", "has scales"])

    assert len(response.results) == EXPECTED_MULTI_RESULTS
    assert response.results[0].code == "03"
    assert response.results[0].distance == pytest.approx(EXPECTED_TOP_DISTANCE)
    fake_store.search_many.assert_called_once()
    assert set(fake_store.search_many.call_args.kwargs["queries"]) == {
        "has gills",
        "has gills has scales",
    }


def test_search_index_multi_filters_none_values(tmp_path: Path) -> None:
    fake_store = SimpleNamespace(
        num_vectors=1,
        search_many=MagicMock(
            return_value=[
                [{"doc_text": "fish", "score": 0.7, "doc_label": "03"}],
            ]
        ),
    )

    with patch(
        "survey_assist_embed_core.embed.embedding."
        "EmbeddingHandler._load_existing_vector_store",
        return_value=(fake_store, "mock-source.csv"),
    ):
        handler = EmbeddingHandler(db_dir=str(tmp_path / "vector_store"))

    with patch.object(handler, "spell", side_effect=lambda text: text):
        response = handler.search_index_multi([None, "has gills"])

    assert len(response.results) == 1
    assert response.results[0].code == "03"
    assert response.results[0].title == "fish"
    assert response.results[0].distance == pytest.approx(0.3)
    fake_store.search_many.assert_called_once_with(queries=["has gills"], limit=1)


def test_search_index_multi_all_none_returns_empty(
    embedding_handler_for_embed: EmbeddingHandler,
) -> None:
    response = embedding_handler_for_embed.search_index_multi([None, None])

    assert isinstance(response, SearchIndexResponse)
    assert response.results == []


def test_embedding_handler_initialization(tmp_path: Path) -> None:
    mock_vector_store = SimpleNamespace(num_vectors=123)
    backend = ClassifaiVectorBackend()
    backend._set_embedding_model_name("other")

    with patch(
        "survey_assist_embed_core.embed.embedding."
        "EmbeddingHandler._load_existing_vector_store",
        return_value=(mock_vector_store, "mock-source.csv"),
    ):
        handler = EmbeddingHandler(
            db_dir=str(tmp_path / "vector_store"),
            backend=backend,
        )

    assert isinstance(handler._backend, ClassifaiVectorBackend)
    assert (
        handler.get_embed_config().backend.settings["embedding_model_name"] == "other"
    )


def test_load_existing_vector_store_local(tmp_path: Path) -> None:
    db_dir = tmp_path / "vector_store"
    db_dir.mkdir()
    (db_dir / "metadata.json").write_text(
        json.dumps({"index_source_file": "source.csv"}),
        encoding="utf-8",
    )
    (db_dir / "vectors.parquet").write_text("dummy", encoding="utf-8")

    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(db_dir)
    handler._backend = SimpleNamespace(load=MagicMock())

    fake_store = SimpleNamespace(num_vectors=42)
    handler._backend.load.return_value = (fake_store, "source.csv")

    result = handler._load_existing_vector_store()

    assert result == (fake_store, "source.csv")
    handler._backend.load.assert_called_once_with(
        folder_path=str(db_dir),
    )


def test_load_existing_vector_store_local_missing_files(tmp_path: Path) -> None:
    db_dir = tmp_path / "vector_store"
    db_dir.mkdir()
    (db_dir / "metadata.json").write_text("{}", encoding="utf-8")

    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(db_dir)
    handler._backend = SimpleNamespace(
        load=MagicMock(side_effect=FileNotFoundError("No persisted vector store found"))
    )

    with pytest.raises(FileNotFoundError, match="No persisted vector store found"):
        handler._load_existing_vector_store()


def test_load_existing_vector_store_error_uses_artifact_store_names(
    tmp_path: Path,
) -> None:
    db_dir = tmp_path / "vector_store"
    db_dir.mkdir()

    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(db_dir)
    handler._backend = SimpleNamespace(
        load=MagicMock(
            side_effect=FileNotFoundError(
                "No persisted vector store found in path. Required persisted artifacts: "
                "store-metadata.json, store-vectors.parquet."
            )
        )
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            r"No persisted vector store found .*"
            r"store-metadata\.json, store-vectors\.parquet.*"
            r"Build the vector-store artifacts before initialising "
            r"EmbeddingHandler\."
        ),
    ):
        handler._load_existing_vector_store()


def test_load_existing_vector_store_gcs() -> None:
    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = "gs://my-bucket/prefix"
    handler._backend = SimpleNamespace(load=MagicMock())

    fake_store = SimpleNamespace(num_vectors=55)
    handler._backend.load.return_value = (fake_store, None)

    with tempfile.TemporaryDirectory() as temp_dir:
        Path(temp_dir, "metadata.json").write_text("{}", encoding="utf-8")
        Path(temp_dir, "vectors.parquet").write_text("dummy", encoding="utf-8")
        downloaded = DownloadedVectorStore(
            path=temp_dir,
            temp_dir=SimpleNamespace(name=temp_dir, cleanup=lambda: None),
        )

        with patch(
            "survey_assist_embed_core.embed.embedding.download_vector_store_from_gcs",
            return_value=downloaded,
        ) as mock_download:
            result = handler._load_existing_vector_store()

    assert result == (fake_store, None)
    mock_download.assert_called_once_with("gs://my-bucket/prefix")
    handler._backend.load.assert_called_once_with(
        folder_path=downloaded.path,
    )


def test_load_existing_vector_store_raises_when_dir_missing(tmp_path: Path) -> None:
    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(tmp_path / "nonexistent")
    handler._backend = SimpleNamespace(
        load=MagicMock(side_effect=FileNotFoundError("No persisted vector store found"))
    )

    with pytest.raises(FileNotFoundError, match="No persisted vector store found"):
        handler._load_existing_vector_store()


def test_load_existing_vector_store_gcs_missing_files_uses_artifact_store_error() -> (
    None
):
    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = "gs://my-bucket/prefix"
    handler._backend = SimpleNamespace(
        load=MagicMock(
            side_effect=FileNotFoundError(
                "No persisted vector store found in path. "
                "Required persisted artifacts: metadata.json, vectors.parquet."
            )
        )
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        Path(temp_dir, "metadata.json").write_text("{}", encoding="utf-8")
        downloaded = DownloadedVectorStore(
            path=temp_dir,
            temp_dir=SimpleNamespace(name=temp_dir, cleanup=lambda: None),
        )

        with (
            patch(
                "survey_assist_embed_core.embed.embedding.download_vector_store_from_gcs",
                return_value=downloaded,
            ),
            pytest.raises(
                FileNotFoundError,
                match=(
                    r"Required persisted artifacts: metadata\.json, vectors\.parquet.*"
                    r"Build the vector-store artifacts before initialising "
                    r"EmbeddingHandler\."
                ),
            ),
        ):
            handler._load_existing_vector_store()


def test_get_embed_config_returns_correct_values(tmp_path: Path) -> None:
    store = SimpleNamespace(num_vectors=7)
    backend = ClassifaiVectorBackend()
    backend._set_embedding_model_name("other")

    with patch(
        "survey_assist_embed_core.embed.embedding."
        "EmbeddingHandler._load_existing_vector_store",
        return_value=(store, "mock-source.csv"),
    ):
        handler = EmbeddingHandler(
            db_dir=str(tmp_path / "vector_store"),
            k_matches=5,
            backend=backend,
        )

    cfg = handler.get_embed_config()

    assert cfg.db_dir == str(tmp_path / "vector_store")
    assert cfg.k_matches == EXPECTED_CONFIG_K_MATCHES
    assert cfg.index_size == EXPECTED_CONFIG_INDEX_SIZE
    assert cfg.backend.backend_name == "classifai"
    assert cfg.backend.settings == {"embedding_model_name": "other"}


def test_normalised_hf_vectoriser_normalize_unit_vectors() -> None:
    inst = NormalisedHFVectoriser.__new__(NormalisedHFVectoriser)
    vectors = np.array([[3.0, 4.0], [1.0, 0.0]])
    result = inst._normalize(vectors)

    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_normalised_hf_vectoriser_zero_vectors_do_not_divide_by_zero() -> None:
    inst = NormalisedHFVectoriser.__new__(NormalisedHFVectoriser)
    vectors = np.array([[0.0, 0.0], [1.0, 0.0]])
    result = inst._normalize(vectors)

    assert np.allclose(result[0], [0.0, 0.0])
    assert np.allclose(np.linalg.norm(result[1]), 1.0)


def test_normalised_hf_vectoriser_transform_single_string_wraps_in_list() -> None:
    inst = NormalisedHFVectoriser.__new__(NormalisedHFVectoriser)
    fake_vec = np.array([[1.0, 0.0]])

    with patch(
        "survey_assist_embed_core.adapters.classifai.vectoriser.HuggingFaceVectoriser.transform",
        return_value=fake_vec,
    ) as mock_super:
        result = inst.transform("hello")

    mock_super.assert_called_once_with(["hello"])
    assert result.shape == (1, 2)


def test_normalised_hf_vectoriser_transform_list_passes_through() -> None:
    inst = NormalisedHFVectoriser.__new__(NormalisedHFVectoriser)
    fake_vec = np.array([[1.0, 0.0], [0.0, 1.0]])

    with patch(
        "survey_assist_embed_core.adapters.classifai.vectoriser.HuggingFaceVectoriser.transform",
        return_value=fake_vec,
    ) as mock_super:
        result = inst.transform(["hello", "world"])

    mock_super.assert_called_once_with(["hello", "world"])
    assert result.shape == (2, 2)
