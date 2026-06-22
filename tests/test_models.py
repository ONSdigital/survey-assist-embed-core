"""Tests for retrieval-facing shared models."""

import pytest

from survey_assist_embed_core.models import (
    EmbeddingStatus,
    SearchIndexItem,
    SearchIndexResponse,
)

READY_INDEX_SIZE = 100
VALID_DB_DIR = "vector-store-dir"


def test_search_index_response_round_trips_items() -> None:
    """Search results should retain the typed payload structure."""
    response = SearchIndexResponse(
        results=[SearchIndexItem(code="01110", title="Growing wheat", distance=0.1)]
    )

    assert response.results[0].code == "01110"
    assert response.results[0].title == "Growing wheat"
    assert response.results[0].distance == pytest.approx(0.1)


def test_embedding_status_valid_ready_state() -> None:
    """A ready status should accept valid retrieval metadata."""
    status = EmbeddingStatus(
        embedding_model_name="all-MiniLM-L6-v2",
        db_dir=VALID_DB_DIR,
        k_matches=10,
        index_source_file="source.csv",
        status="ready",
        index_size=READY_INDEX_SIZE,
    )

    assert status.status == "ready"
    assert status.index_size == READY_INDEX_SIZE


def test_embedding_status_rejects_zero_index_size() -> None:
    """A ready status must report at least one indexed record."""
    with pytest.raises(ValueError, match="index_size must be at least 1"):
        EmbeddingStatus(
            embedding_model_name="all-MiniLM-L6-v2",
            db_dir=VALID_DB_DIR,
            k_matches=10,
            index_source_file="source.csv",
            status="ready",
            index_size=0,
        )


def test_embedding_status_rejects_empty_model_name() -> None:
    """A ready status should reject an empty model name."""
    with pytest.raises(ValueError, match="embedding_model_name must be a valid value"):
        EmbeddingStatus(
            embedding_model_name="",
            db_dir=VALID_DB_DIR,
            k_matches=10,
            index_source_file="source.csv",
            status="ready",
            index_size=5,
        )


def test_embedding_status_rejects_unknown_db_dir() -> None:
    """A ready status should reject placeholder db_dir values."""
    with pytest.raises(ValueError, match="db_dir must be a valid value"):
        EmbeddingStatus(
            embedding_model_name="all-MiniLM-L6-v2",
            db_dir="unknown",
            k_matches=10,
            index_source_file="source.csv",
            status="ready",
            index_size=5,
        )
