"""SIC-specific helpers for building retrieval indexes from published files."""

import logging
import tempfile
from pathlib import Path
from typing import TypedDict, Unpack

from survey_assist_embed_core.adapters.taxonomy import (
    SICFileReference,
    build_sic_leaf_text_frame,
    load_sic_hierarchy_from_index_files,
)
from survey_assist_embed_core.embed.embedding import DEFAULT_DB_DIR, EmbeddingHandler

logger = logging.getLogger(__name__)


class _EmbeddingHandlerKwargs(TypedDict, total=False):
    embedding_model_name: str
    k_matches: int


def load_embedding_handler_from_sic_index_files(
    *,
    sic_index_file: SICFileReference,
    sic_structure_file: SICFileReference,
    db_dir: str = DEFAULT_DB_DIR,
    **kwargs: Unpack[_EmbeddingHandlerKwargs],
) -> EmbeddingHandler:
    """Build an embedding handler from published SIC index and structure files."""
    sic = load_sic_hierarchy_from_index_files(
        sic_index_file=sic_index_file,
        sic_structure_file=sic_structure_file,
    )
    leaf_text = build_sic_leaf_text_frame(sic)

    csv_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            suffix=".csv",
        ) as temp_csv:
            csv_path = temp_csv.name

        leaf_text.to_csv(csv_path, index=False)
        logger.info("Temporary SIC CSV for vector store created at %s", csv_path)

        return EmbeddingHandler(
            db_dir=db_dir,
            index_source_file=csv_path,
            **kwargs,
        )
    finally:
        if csv_path is not None:
            Path(csv_path).unlink(missing_ok=True)
