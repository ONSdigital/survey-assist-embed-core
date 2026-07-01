"""Tests for retrieval-facing shared models."""

import pytest

from survey_assist_embed_core.models import (
    EmbeddingStatus,
    SearchIndexItem,
    SearchIndexResponse,
    VectorBackendConfig,
)
from survey_assist_embed_core.models.status import MAX_K_MATCHES

READY_INDEX_SIZE = 100
VALID_DB_DIR = "vector-store-dir"


def make_backend_config(model_name: str = "all-MiniLM-L6-v2") -> VectorBackendConfig:
    """Build a valid backend config payload for status tests."""
    return VectorBackendConfig(
        backend_name="classifai",
        settings={"embedding_model_name": model_name},
    )


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
        db_dir=VALID_DB_DIR,
        k_matches=10,
        index_source_file="source.csv",
        backend=make_backend_config(),
        status="ready",
        index_size=READY_INDEX_SIZE,
    )

    assert status.status == "ready"
    assert status.index_size == READY_INDEX_SIZE
    assert status.backend.settings["embedding_model_name"] == "all-MiniLM-L6-v2"


def test_embedding_status_rejects_zero_index_size() -> None:
    """A ready status must report at least one indexed record."""
    with pytest.raises(ValueError, match="index_size must be at least 1"):
        EmbeddingStatus(
            db_dir=VALID_DB_DIR,
            k_matches=10,
            index_source_file="source.csv",
            backend=make_backend_config(),
            status="ready",
            index_size=0,
        )


def test_embedding_status_rejects_empty_backend_name() -> None:
    """Status should reject an empty backend name."""
    with pytest.raises(ValueError, match="backend_name must be a valid value"):
        EmbeddingStatus(
            db_dir=VALID_DB_DIR,
            k_matches=10,
            index_source_file="source.csv",
            backend=VectorBackendConfig(backend_name="", settings={}),
            status="ready",
            index_size=5,
        )


def test_embedding_status_rejects_unknown_db_dir() -> None:
    """A ready status should reject placeholder db_dir values."""
    with pytest.raises(ValueError, match="db_dir must be a valid value"):
        EmbeddingStatus(
            db_dir="unknown",
            k_matches=10,
            index_source_file="source.csv",
            backend=make_backend_config(),
            status="ready",
            index_size=5,
        )


def test_embedding_status_rejects_zero_k_matches() -> None:
    """The retrieval config should reject k_matches below one."""
    with pytest.raises(ValueError, match="k_matches must be at least 1"):
        EmbeddingStatus(
            db_dir=VALID_DB_DIR,
            k_matches=0,
            index_source_file="source.csv",
            backend=make_backend_config(),
            status="loading",
            index_size=0,
        )


def test_embedding_status_rejects_k_matches_above_maximum() -> None:
    """The retrieval config should reject k_matches above the supported bound."""
    with pytest.raises(ValueError, match=rf"k_matches must be at most {MAX_K_MATCHES}"):
        EmbeddingStatus(
            db_dir=VALID_DB_DIR,
            k_matches=MAX_K_MATCHES + 1,
            index_source_file="source.csv",
            backend=make_backend_config(),
            status="loading",
            index_size=0,
        )
