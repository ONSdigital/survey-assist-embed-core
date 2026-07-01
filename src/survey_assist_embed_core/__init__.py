"""Shared retrieval primitives for Survey Assist embedding and search."""

from survey_assist_embed_core.adapters.classifai.vector_backend import (
    ClassifaiVectorBackend,
)
from survey_assist_embed_core.adapters.classifai.vector_backend import (
    build_classifai_vector_store_artifacts as build_embedding_index,
)
from survey_assist_embed_core.embed.embedding import EmbeddingHandler

__all__ = [
    "ClassifaiVectorBackend",
    "EmbeddingHandler",
    "build_embedding_index",
]
