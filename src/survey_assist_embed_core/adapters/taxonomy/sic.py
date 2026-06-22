"""SIC taxonomy adapters for published structure and index source files."""

import logging
import os
from os import PathLike
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from industrial_classification.hierarchy.sic_hierarchy import (  # type: ignore[import-untyped]
    SIC,
    load_hierarchy,
)

logger = logging.getLogger(__name__)

type SICFileReference = str | PathLike[str] | tuple[str | PathLike[str], str]

_SIC_INDEX_DTYPES = {
    "uk_sic_2007": "string",
    "activity": "string",
}

_SIC_STRUCTURE_DTYPES = {
    "description": "string",
    "section": "string",
    "most_disaggregated_level": "string",
    "level_headings": "string",
}


def _resolve_sic_file_reference(
    file_reference: SICFileReference,
) -> tuple[str, str | None]:
    """Resolve a SIC file reference into a path and optional file-type hint."""
    if isinstance(file_reference, tuple):
        path, file_type = file_reference
        return os.fspath(path), file_type.removeprefix(".").lower()

    return os.fspath(file_reference), None


def _read_sic_table(
    file_reference: SICFileReference,
    *,
    dtype: dict[str, str],
) -> pd.DataFrame:
    """Read a SIC source table from CSV or Excel while preserving key codes."""
    file_path, file_type_hint = _resolve_sic_file_reference(file_reference)
    suffix = Path(file_path).suffix.removeprefix(".").lower()
    file_type = suffix or file_type_hint

    logger.info("Reading SIC source table from %s", file_path)

    if file_type in {"xls", "xlsx"}:
        return pd.read_excel(file_path, dtype=dtype)

    if file_type == "csv":
        return pd.read_csv(file_path, dtype=dtype)

    raise ValueError(
        "Unsupported SIC source file type: "
        f"{file_type!r}. Expected one of 'csv', 'xls', or 'xlsx'."
    )


def load_sic_hierarchy_from_index_files(
    *,
    sic_index_file: SICFileReference,
    sic_structure_file: SICFileReference,
) -> SIC:
    """Load the SIC hierarchy from published index and structure source files."""
    logger.info(
        "Loading SIC hierarchy from index=%s structure=%s",
        sic_index_file,
        sic_structure_file,
    )

    sic_index_df = _read_sic_table(sic_index_file, dtype=_SIC_INDEX_DTYPES)
    sic_structure_df = _read_sic_table(
        sic_structure_file,
        dtype=_SIC_STRUCTURE_DTYPES,
    )
    return load_hierarchy(sic_structure_df, sic_index_df)


def build_sic_leaf_text_frame(sic: SIC) -> pd.DataFrame:
    """Build the SIC leaf-text index used by the shared embedding handler."""
    leaf_text = sic.all_leaf_text().copy()
    cleaned_codes = (
        leaf_text["code"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace("/", "", regex=False)
    )
    leaf_text["label"] = (cleaned_codes + "0").str[:5]
    return leaf_text
