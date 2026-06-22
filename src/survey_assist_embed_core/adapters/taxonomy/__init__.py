"""Taxonomy adapters for retrieval-oriented embedding workflows."""

from survey_assist_embed_core.adapters.taxonomy.sic import (
    SICFileReference,
    build_sic_leaf_text_frame,
    load_sic_hierarchy_from_index_files,
)

__all__ = [
    "SICFileReference",
    "build_sic_leaf_text_frame",
    "load_sic_hierarchy_from_index_files",
]
