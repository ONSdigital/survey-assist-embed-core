# pylint: disable=missing-function-docstring
"""Tests for SIC-specific embedding compatibility helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from survey_assist_embed_core.adapters.taxonomy.sic import (
    load_sic_hierarchy_from_index_files,
)
from survey_assist_embed_core.embed import load_embedding_handler_from_sic_index_files

EXPECTED_K_MATCHES = 42


class FakeSicHierarchy:  # pylint: disable=too-few-public-methods
    """Minimal SIC hierarchy stub exposing leaf-text data."""

    def __init__(self, leaf_text: pd.DataFrame):
        self._leaf_text = leaf_text

    def all_leaf_text(self) -> pd.DataFrame:
        return self._leaf_text.copy()


def test_load_sic_hierarchy_from_index_files_reads_expected_tables(tmp_path: Path):
    sic_index_path = tmp_path / "sic-index.xlsx"
    sic_structure_path = tmp_path / "sic-structure.csv"

    sic_index_df = pd.DataFrame(
        {"uk_sic_2007": ["01110"], "activity": ["Growing of cereals"]}
    )
    sic_structure_df = pd.DataFrame(
        {
            "description": ["Growing of cereals"],
            "section": ["A"],
            "most_disaggregated_level": ["0111"],
            "level_headings": ["class"],
        }
    )
    sentinel_sic = object()

    with (
        patch(
            "survey_assist_embed_core.adapters.taxonomy.sic.pd.read_excel",
            return_value=sic_index_df,
        ) as mock_read_excel,
        patch(
            "survey_assist_embed_core.adapters.taxonomy.sic.pd.read_csv",
            return_value=sic_structure_df,
        ) as mock_read_csv,
        patch(
            "survey_assist_embed_core.adapters.taxonomy.sic.load_hierarchy",
            return_value=sentinel_sic,
        ) as mock_load_hierarchy,
    ):
        result = load_sic_hierarchy_from_index_files(
            sic_index_file=(sic_index_path, "xlsx"),
            sic_structure_file=(sic_structure_path, "csv"),
        )

    assert result is sentinel_sic
    assert mock_read_excel.call_args.kwargs["dtype"]["uk_sic_2007"] == "string"
    assert mock_read_csv.call_args.kwargs["dtype"]["level_headings"] == "string"

    structure_arg, index_arg = mock_load_hierarchy.call_args.args
    pd.testing.assert_frame_equal(structure_arg, sic_structure_df)
    pd.testing.assert_frame_equal(index_arg, sic_index_df)


def test_load_embedding_handler_from_sic_index_files_builds_temp_csv(
    tmp_path: Path,
):
    leaf_text = pd.DataFrame(
        {
            "code": ["01.11", "01/12"],
            "text": ["alpha", "beta"],
        }
    )
    fake_sic = FakeSicHierarchy(leaf_text)
    built_handler = SimpleNamespace()
    captured_frame: pd.DataFrame | None = None

    def fake_embedding_handler(**kwargs: object) -> SimpleNamespace:
        nonlocal captured_frame
        csv_path = kwargs["index_source_file"]
        assert isinstance(csv_path, str)
        captured_frame = pd.read_csv(
            csv_path,
            dtype={"code": "string", "text": "string", "label": "string"},
        )
        return built_handler

    with (
        patch(
            "survey_assist_embed_core.embed.sic_specific_embed."
            "load_sic_hierarchy_from_index_files",
            return_value=fake_sic,
        ) as mock_load_sic,
        patch(
            "survey_assist_embed_core.embed.sic_specific_embed.EmbeddingHandler",
            side_effect=fake_embedding_handler,
        ) as mock_handler_cls,
    ):
        result = load_embedding_handler_from_sic_index_files(
            db_dir=str(tmp_path / "vector_store"),
            sic_index_file="sic-index.xlsx",
            sic_structure_file="sic-structure.xlsx",
        )

    assert result is built_handler
    mock_load_sic.assert_called_once_with(
        sic_index_file="sic-index.xlsx",
        sic_structure_file="sic-structure.xlsx",
    )

    assert captured_frame is not None
    assert captured_frame["code"].tolist() == ["01.11", "01/12"]
    assert captured_frame["label"].tolist() == ["01110", "01120"]

    index_source_file = mock_handler_cls.call_args.kwargs["index_source_file"]
    assert isinstance(index_source_file, str)
    assert not Path(index_source_file).exists()


def test_load_embedding_handler_from_sic_index_files_forwards_kwargs(
    tmp_path: Path,
):
    fake_sic = FakeSicHierarchy(pd.DataFrame({"code": ["01110"], "text": ["alpha"]}))
    built_handler = SimpleNamespace()

    with (
        patch(
            "survey_assist_embed_core.embed.sic_specific_embed."
            "load_sic_hierarchy_from_index_files",
            return_value=fake_sic,
        ),
        patch(
            "survey_assist_embed_core.embed.sic_specific_embed.EmbeddingHandler",
            return_value=built_handler,
        ) as mock_handler_cls,
    ):
        load_embedding_handler_from_sic_index_files(
            db_dir=str(tmp_path / "vector_store"),
            sic_index_file="sic-index.xlsx",
            sic_structure_file="sic-structure.xlsx",
            embedding_model_name="other",
            k_matches=EXPECTED_K_MATCHES,
        )

    call_kwargs = mock_handler_cls.call_args.kwargs
    assert call_kwargs["db_dir"] == str(tmp_path / "vector_store")
    assert call_kwargs["embedding_model_name"] == "other"
    assert call_kwargs["k_matches"] == EXPECTED_K_MATCHES
