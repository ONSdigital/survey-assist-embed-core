"""Retrieval configuration and status models."""

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_K_MATCHES = 100

type BackendConfigValue = str | int | float | bool | None


class VectorBackendConfig(BaseModel):
    """Configuration metadata for a concrete vector backend.

    Attributes:
        backend_name: Stable name for the backend implementation.
        settings: Backend-specific configuration values safe to expose in
            status output.
    """

    backend_name: str
    settings: dict[str, BackendConfigValue] = Field(default_factory=dict)

    @field_validator("backend_name")
    @classmethod
    def _validate_backend_name(cls, value: str) -> str:
        if not value or value == "unknown":
            raise ValueError("backend_name must be a valid value")
        return value


class EmbeddingConfig(BaseModel):
    """Configuration for retrieval against a persisted vector store.

    Attributes:
        db_dir: Local directory or GCS URI for the persisted vector store.
        index_source_file: Recorded source-file path for the loaded index, if
            available.
        k_matches: Maximum number of matches returned per search.
        backend: Metadata for the backend used to load and query the index.
    """

    db_dir: str
    index_source_file: str | None = None
    k_matches: int
    backend: VectorBackendConfig

    @field_validator("k_matches")
    @classmethod
    def _validate_k_matches(cls, value: int) -> int:
        if value < 1:
            raise ValueError("k_matches must be at least 1")
        if value > MAX_K_MATCHES:
            raise ValueError(f"k_matches must be at most {MAX_K_MATCHES}")
        return value


class EmbeddingStatus(EmbeddingConfig):
    """Runtime status for a loaded embedding index.

    Attributes:
        status: Lifecycle state for the handler, such as ``ready``.
        index_size: Number of vectors available in the loaded index.
    """

    status: str
    index_size: int

    @model_validator(mode="after")
    def _validate_ready_state(self) -> "EmbeddingStatus":
        if self.status == "ready":
            if self.index_size < 1:
                raise ValueError("index_size must be at least 1 when ready")
            if not self.db_dir or self.db_dir == "unknown":
                raise ValueError("db_dir must be a valid value when ready")
        return self
