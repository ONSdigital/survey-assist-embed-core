"""Retrieval configuration and status models."""

from pydantic import BaseModel, field_validator, model_validator

MAX_K_MATCHES = 100


class EmbeddingConfig(BaseModel):
    """Configuration for an embedding model and vector store."""

    embedding_model_name: str
    db_dir: str
    index_source_file: str | None = None
    k_matches: int

    @field_validator("k_matches")
    @classmethod
    def _validate_k_matches(cls, value: int) -> int:
        if value < 1:
            raise ValueError("k_matches must be at least 1")
        if value > MAX_K_MATCHES:
            raise ValueError(f"k_matches must be at most {MAX_K_MATCHES}")
        return value


class EmbeddingStatus(EmbeddingConfig):
    """Runtime status for a ready or loading embedding index."""

    status: str
    index_size: int

    @model_validator(mode="after")
    def _validate_ready_state(self) -> "EmbeddingStatus":
        if self.status == "ready":
            if self.index_size < 1:
                raise ValueError("index_size must be at least 1 when ready")
            for field_name, field_value in [
                ("embedding_model_name", self.embedding_model_name),
                ("db_dir", self.db_dir),
            ]:
                if not field_value or field_value == "unknown":
                    raise ValueError(f"{field_name} must be a valid value when ready")
        return self
