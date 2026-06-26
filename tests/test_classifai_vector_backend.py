# pylint: disable=missing-function-docstring
"""Tests for the ClassifAI vector backend adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from classifai.indexers import VectorStoreSearchOutput

from survey_assist_embed_core.adapters.classifai import ClassifaiVectorBackend

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
    backend = ClassifaiVectorBackend(embedding_model_name="other")
    vectoriser = object()
    folder_path = str(tmp_path / "vector_store")
    fake_store = SimpleNamespace(
        num_vectors=EXPECTED_LOADED_VECTOR_COUNT,
        search=MagicMock(),
    )

    with (
        patch.object(
            backend,
            "_build_vectoriser",
            return_value=vectoriser,
        ) as mock_build_vectoriser,
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "VectorStore.from_filespace",
            return_value=fake_store,
        ) as mock_from_filespace,
    ):
        index = backend.load(folder_path=folder_path)

    assert index.num_vectors == EXPECTED_LOADED_VECTOR_COUNT
    mock_build_vectoriser.assert_called_once_with()
    mock_from_filespace.assert_called_once_with(
        folder_path=folder_path,
        vectoriser=vectoriser,
        hooks=None,
    )


def test_classifai_vector_backend_build_uses_expected_args() -> None:
    backend = ClassifaiVectorBackend(embedding_model_name="other")
    vectoriser = object()
    fake_store = SimpleNamespace(
        num_vectors=EXPECTED_BUILT_VECTOR_COUNT,
        search=MagicMock(),
    )

    with (
        patch.object(
            backend,
            "_build_vectoriser",
            return_value=vectoriser,
        ) as mock_build_vectoriser,
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend.VectorStore",
            return_value=fake_store,
        ) as mock_vector_store,
    ):
        index = backend.build(
            file_name="source.csv",
            output_dir="vector_store",
        )

    assert index.num_vectors == EXPECTED_BUILT_VECTOR_COUNT
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


def test_classifai_vector_backend_search_returns_records(tmp_path) -> None:
    rows = [{"doc_text": "dog", "score": EXPECTED_SEARCH_SCORE, "doc_label": "02"}]
    folder_path = str(tmp_path / "vector_store")
    fake_store = SimpleNamespace(
        num_vectors=1,
        search=MagicMock(return_value=make_search_output(rows)),
    )
    backend = ClassifaiVectorBackend(embedding_model_name="other")

    with (
        patch.object(
            backend,
            "_build_vectoriser",
            return_value=object(),
        ),
        patch(
            "survey_assist_embed_core.adapters.classifai.vector_backend."
            "VectorStore.from_filespace",
            return_value=fake_store,
        ),
    ):
        index = backend.load(folder_path=folder_path)

    results = index.search("dog", limit=EXPECTED_SEARCH_LIMIT)

    assert fake_store.search.call_args.kwargs["n_results"] == EXPECTED_SEARCH_LIMIT
    assert results[0]["doc_label"] == "02"
    assert results[0]["doc_text"] == "dog"
    assert results[0]["score"] == EXPECTED_SEARCH_SCORE
