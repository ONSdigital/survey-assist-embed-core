# pylint: disable=missing-function-docstring, protected-access, redefined-outer-name
"""Tests for the shared retrieval embedding handler and vectoriser."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from classifai.indexers import VectorStoreSearchOutput

from survey_assist_embed_core.adapters.storage.gcs import DownloadedVectorStore
from survey_assist_embed_core.embed import EmbeddingHandler
from survey_assist_embed_core.embed.embedding import ChromaDBesqueHFVectoriser
from survey_assist_embed_core.models import SearchIndexItem, SearchIndexResponse

EXPECTED_TOY_INDEX_SIZE = 4
EXPECTED_MULTI_RESULTS = 4
EXPECTED_TOP_DISTANCE = 0.1
EXPECTED_CONFIG_K_MATCHES = 5
EXPECTED_CONFIG_INDEX_SIZE = 7


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


@pytest.fixture
def embedding_handler_for_embed(tmp_path: Path) -> EmbeddingHandler:
    placeholder_store = SimpleNamespace(num_vectors=1)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "survey_assist_embed_core.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "survey_assist_embed_core.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(placeholder_store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

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
        search=MagicMock(return_value=make_search_output(rows)),
    )
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "survey_assist_embed_core.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "survey_assist_embed_core.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(fake_store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    return handler


def test_embedding_handler_init_sets_vector_store(tmp_path: Path) -> None:
    built_store = SimpleNamespace(num_vectors=EXPECTED_TOY_INDEX_SIZE)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "survey_assist_embed_core.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "survey_assist_embed_core.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(built_store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    assert handler.vector_store is built_store


def test_embedding_handler_init_sets_index_size(tmp_path: Path) -> None:
    built_store = SimpleNamespace(num_vectors=EXPECTED_TOY_INDEX_SIZE)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "survey_assist_embed_core.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "survey_assist_embed_core.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(built_store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    assert handler.index_size == EXPECTED_TOY_INDEX_SIZE


def test_search_index(embedding_handler_search: EmbeddingHandler) -> None:
    response = embedding_handler_search.search_index("mens best friend")

    assert isinstance(response, SearchIndexResponse)
    assert response.results[0].code == "02"
    assert response.results[0].title == "dog"
    assert response.results[0].distance == pytest.approx(0.01)


def test_search_index_uses_vector_store_search_output(tmp_path: Path) -> None:
    rows = [{"doc_text": "dog", "score": 0.9, "doc_label": "02"}]
    fake_store = SimpleNamespace(
        num_vectors=1,
        search=MagicMock(return_value=make_search_output(rows)),
    )
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "survey_assist_embed_core.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "survey_assist_embed_core.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(fake_store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    results = handler.search_index("dog")

    assert isinstance(results, SearchIndexResponse)
    assert results.results[0].code == "02"
    assert results.results[0].title == "dog"


def test_search_index_multi(tmp_path: Path) -> None:
    placeholder_store = SimpleNamespace(num_vectors=1)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "survey_assist_embed_core.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "survey_assist_embed_core.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(placeholder_store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    with (
        patch.object(handler, "spell", side_effect=lambda text: text),
        patch.object(
            handler,
            "search_index",
            side_effect=[
                SearchIndexResponse(
                    results=[
                        SearchIndexItem(code="03", title="fish", distance=0.4),
                        SearchIndexItem(code="04", title="lizard", distance=0.6),
                    ]
                ),
                SearchIndexResponse(
                    results=[
                        SearchIndexItem(code="03", title="fish", distance=0.1),
                        SearchIndexItem(code="04", title="lizard", distance=0.2),
                    ]
                ),
            ],
        ),
    ):
        response = handler.search_index_multi(["has gills", "has scales"])

    assert len(response.results) == EXPECTED_MULTI_RESULTS
    assert response.results[0].code == "03"
    assert response.results[0].distance == EXPECTED_TOP_DISTANCE


def test_search_index_multi_filters_none_values(tmp_path: Path) -> None:
    placeholder_store = SimpleNamespace(num_vectors=1)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "survey_assist_embed_core.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "survey_assist_embed_core.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(placeholder_store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    with (
        patch.object(handler, "spell", side_effect=lambda text: text),
        patch.object(
            handler,
            "search_index",
            return_value=SearchIndexResponse(
                results=[SearchIndexItem(code="03", title="fish", distance=0.3)]
            ),
        ) as mock_search,
    ):
        response = handler.search_index_multi([None, "has gills"])

    assert response.results == [SearchIndexItem(code="03", title="fish", distance=0.3)]
    mock_search.assert_called_once_with(query="has gills")


def test_search_index_multi_all_none_returns_empty(
    embedding_handler_for_embed: EmbeddingHandler,
) -> None:
    response = embedding_handler_for_embed.search_index_multi([None, None])

    assert isinstance(response, SearchIndexResponse)
    assert response.results == []


def test_embedding_handler_initialization(tmp_path: Path) -> None:
    mock_vector_store = SimpleNamespace(num_vectors=123)

    with (
        patch(
            "survey_assist_embed_core.embed.embedding.ChromaDBesqueHFVectoriser"
        ) as mock_hf,
        patch(
            "survey_assist_embed_core.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(mock_vector_store, "mock-source.csv"),
        ),
    ):
        EmbeddingHandler("other", db_dir=str(tmp_path / "vector_store"))

        mock_hf.assert_called_once_with(model_name="sentence-transformers/other")


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
    handler.embeddings = object()
    handler._downloaded_vector_store = None

    fake_store = SimpleNamespace(num_vectors=42)

    with (
        patch(
            "survey_assist_embed_core.embed.embedding.is_gcs_path",
            return_value=False,
        ),
        patch(
            "survey_assist_embed_core.embed.embedding.VectorStore.from_filespace",
            return_value=fake_store,
        ) as mock_from_filespace,
    ):
        result = handler._load_existing_vector_store()

    assert result == (fake_store, "source.csv")
    mock_from_filespace.assert_called_once_with(
        folder_path=str(db_dir),
        vectoriser=handler.embeddings,
        hooks=None,
    )


def test_load_existing_vector_store_local_missing_files(tmp_path: Path) -> None:
    db_dir = tmp_path / "vector_store"
    db_dir.mkdir()
    (db_dir / "metadata.json").write_text("{}", encoding="utf-8")

    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(db_dir)
    handler.embeddings = object()
    handler._downloaded_vector_store = None

    with (
        patch(
            "survey_assist_embed_core.embed.embedding.is_gcs_path",
            return_value=False,
        ),
        pytest.raises(FileNotFoundError, match="No existing vector store found"),
    ):
        handler._load_existing_vector_store()


def test_load_existing_vector_store_gcs() -> None:
    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = "gs://my-bucket/prefix"
    handler.embeddings = object()
    handler._downloaded_vector_store = None

    fake_store = SimpleNamespace(num_vectors=55)

    with tempfile.TemporaryDirectory() as temp_dir:
        Path(temp_dir, "metadata.json").write_text("{}", encoding="utf-8")
        Path(temp_dir, "vectors.parquet").write_text("dummy", encoding="utf-8")
        downloaded = DownloadedVectorStore(
            path=temp_dir,
            temp_dir=SimpleNamespace(name=temp_dir),
        )

        with (
            patch(
                "survey_assist_embed_core.embed.embedding.is_gcs_path",
                return_value=True,
            ),
            patch(
                "survey_assist_embed_core.embed.embedding.download_vector_store_from_gcs",
                return_value=downloaded,
            ) as mock_download,
            patch(
                "survey_assist_embed_core.embed.embedding.VectorStore.from_filespace",
                return_value=fake_store,
            ) as mock_from_filespace,
        ):
            result = handler._load_existing_vector_store()

    assert result == (fake_store, None)
    assert handler._downloaded_vector_store is downloaded
    mock_download.assert_called_once_with("gs://my-bucket/prefix")
    mock_from_filespace.assert_called_once_with(
        folder_path=downloaded.temp_dir.name,
        vectoriser=handler.embeddings,
        hooks=None,
    )


def test_load_existing_vector_store_raises_when_dir_missing(tmp_path: Path) -> None:
    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(tmp_path / "nonexistent")
    handler.embeddings = object()
    handler._downloaded_vector_store = None

    with (
        patch(
            "survey_assist_embed_core.embed.embedding.is_gcs_path",
            return_value=False,
        ),
        pytest.raises(FileNotFoundError, match="No existing vector store found"),
    ):
        handler._load_existing_vector_store()


def test_build_vector_store_raises_when_db_dir_missing() -> None:
    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = None

    with pytest.raises(ValueError, match="db_dir must be provided"):
        handler._build_vector_store()


def test_build_vector_store_passes_none_metadata(tmp_path: Path) -> None:
    db_dir = tmp_path / "vector_store"
    db_dir.mkdir()
    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(db_dir)
    handler.embeddings = object()
    handler.index_source_file = "some-file.csv"

    built_store = SimpleNamespace(num_vectors=1)

    with patch(
        "survey_assist_embed_core.embed.embedding.VectorStore",
        return_value=built_store,
    ) as mock_vector_store:
        handler._build_vector_store()

    assert mock_vector_store.call_args.kwargs["meta_data"] is None


def test_build_vector_store_uses_downloaded_gcs_source_file(tmp_path: Path) -> None:
    db_dir = tmp_path / "vector_store"
    db_dir.mkdir()
    downloaded_csv = tmp_path / "downloaded.csv"
    downloaded_csv.write_text("doc", encoding="utf-8")

    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(db_dir)
    handler.embeddings = object()
    handler.index_source_file = "gs://my-bucket/data.csv"

    downloaded = DownloadedVectorStore(
        path=str(downloaded_csv),
        temp_dir=SimpleNamespace(name=str(tmp_path)),
    )
    built_store = SimpleNamespace(num_vectors=1)

    with (
        patch(
            "survey_assist_embed_core.embed.embedding.is_gcs_path",
            return_value=True,
        ),
        patch(
            "survey_assist_embed_core.embed.embedding.download_one_file_from_gcs",
            return_value=downloaded,
        ) as mock_download,
        patch(
            "survey_assist_embed_core.embed.embedding.VectorStore",
            return_value=built_store,
        ) as mock_vector_store,
    ):
        handler._build_vector_store()

    mock_download.assert_called_once_with("gs://my-bucket/data.csv")
    assert mock_vector_store.call_args.kwargs["file_name"] == str(downloaded_csv)


def test_build_vector_store_logs_warning_when_parquet_exists(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    db_dir = tmp_path / "vector_store"
    db_dir.mkdir()
    (db_dir / "vectors.parquet").write_text("dummy", encoding="utf-8")

    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(db_dir)
    handler.embeddings = object()
    handler.index_source_file = "some-file.csv"

    built_store = SimpleNamespace(num_vectors=1)

    with (
        caplog.at_level(
            logging.WARNING, logger="survey_assist_embed_core.embed.embedding"
        ),
        patch(
            "survey_assist_embed_core.embed.embedding.VectorStore",
            return_value=built_store,
        ),
    ):
        handler._build_vector_store()

    assert any("overwritten" in record.message for record in caplog.records)


def test_embedding_handler_builds_vector_store_from_source(tmp_path: Path) -> None:
    built_store = SimpleNamespace(num_vectors=2)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "survey_assist_embed_core.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "survey_assist_embed_core.embed.embedding."
            "EmbeddingHandler._build_vector_store",
            return_value=built_store,
        ) as mock_build,
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
            index_source_file="some-source.csv",
        )

    assert handler.vector_store is built_store
    mock_build.assert_called_once()


def test_get_embed_config_returns_correct_values(tmp_path: Path) -> None:
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")
    store = SimpleNamespace(num_vectors=7)

    with (
        patch(
            "survey_assist_embed_core.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "survey_assist_embed_core.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
            k_matches=5,
        )

    cfg = handler.get_embed_config()

    assert cfg.embedding_model_name == "other"
    assert cfg.db_dir == str(tmp_path / "vector_store")
    assert cfg.k_matches == EXPECTED_CONFIG_K_MATCHES
    assert cfg.index_size == EXPECTED_CONFIG_INDEX_SIZE


def test_chromadbesque_normalize_unit_vectors() -> None:
    inst = ChromaDBesqueHFVectoriser.__new__(ChromaDBesqueHFVectoriser)
    vectors = np.array([[3.0, 4.0], [1.0, 0.0]])
    result = inst._normalize(vectors)

    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_chromadbesque_normalize_zero_vectors_do_not_divide_by_zero() -> None:
    inst = ChromaDBesqueHFVectoriser.__new__(ChromaDBesqueHFVectoriser)
    vectors = np.array([[0.0, 0.0], [1.0, 0.0]])
    result = inst._normalize(vectors)

    assert np.allclose(result[0], [0.0, 0.0])
    assert np.allclose(np.linalg.norm(result[1]), 1.0)


def test_chromadbesque_transform_single_string_wraps_in_list() -> None:
    inst = ChromaDBesqueHFVectoriser.__new__(ChromaDBesqueHFVectoriser)
    fake_vec = np.array([[1.0, 0.0]])

    with patch(
        "survey_assist_embed_core.embed.embedding.HuggingFaceVectoriser.transform",
        return_value=fake_vec,
    ) as mock_super:
        result = inst.transform("hello")

    mock_super.assert_called_once_with(["hello"])
    assert result.shape == (1, 2)


def test_chromadbesque_transform_list_passes_through() -> None:
    inst = ChromaDBesqueHFVectoriser.__new__(ChromaDBesqueHFVectoriser)
    fake_vec = np.array([[1.0, 0.0], [0.0, 1.0]])

    with patch(
        "survey_assist_embed_core.embed.embedding.HuggingFaceVectoriser.transform",
        return_value=fake_vec,
    ) as mock_super:
        result = inst.transform(["hello", "world"])

    mock_super.assert_called_once_with(["hello", "world"])
    assert result.shape == (2, 2)
