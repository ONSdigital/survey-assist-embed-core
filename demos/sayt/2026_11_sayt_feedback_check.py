"""Update docstrings before PR."""

# pylint: disable=C0103

# %%
import os
import re

import pandas as pd

from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    NgramWeightSpec,
    PrefixRetrieverSpec,
    PrefixWeightSpec,
    SAYTSuggester,
    SemanticRetrieverSpec,
    SemanticWeightSpec,
    WeightSpecs,
)
from survey_assist_embed_core.sayt.core import _normalise

# %%
MAX_SUGGESTIONS = 20
EXPECTED_CODE_LENGTH = 5
INVALID_VALUES = (
    -1,
    -9,
    -8,
    -10,
    "-1",
    "-9",
    "-8",
    "-10",
    "4+",
    "",
    ".",
    " ",
    None,
    "NAN",
    "NaN",
    "nan",
    "None",
    "Null",
    "<NA>",
)

# %%
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")

# %%
# ENABLE READING FROM THE BUCKET
# lookup_file = f"gs://{bucket_name}/sic_knowledgebase/sic_kb_for_sayt.csv"
lookup_file = "data/sic_kb_for_sayt.csv"
sayt_df = pd.read_csv(lookup_file, dtype=str)


# %%
def parse_numerical_code(
    candidates_str: str,
    code_regex_pattern: str | None = None,
    code_length: int | None = None,
) -> set[str]:
    """Converts the clerical coder responses from a
    stringified list to a proper list of strings.
    E.g. "[8601x, 1410, nan]" -> {'8601x', '01410'}.

    Args:
        candidates_str: String containing the clerical coder responses.
        code_regex_pattern: Regex pattern to extract codes from the string.
        code_length: Number of digits to which the codes should be zero-padded.

    Returns:
        List of cleaned and zero-padded code strings.
    """
    if code_regex_pattern is None:
        code_regex_pattern = r"([0-9]+x*X*)"
    if code_length is None:
        code_length = EXPECTED_CODE_LENGTH

    candidates_str = str(candidates_str).strip()
    if candidates_str in INVALID_VALUES:
        return set()
    try:
        # remove -9 and 4+ from the string
        candidates_str = candidates_str.replace("-9", "").replace("4+", "")
        # Extract all RagCandidate entries using regex
        matches = re.findall(code_regex_pattern, candidates_str)
        return {matches.zfill(code_length) for matches in matches}
    except re.error:
        return set()


def build_sayt_corpus_from_df(
    df: pd.DataFrame,
    search_text_col: str,
    display_text_col: str,
    code_col: str = "code",
    incl_code_in_display: bool = True,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Build a SAYT corpus from a DataFrame.

    Normalises codes using `parse_numerical_code`, optionally appends
    codes to display text, and returns both the updated DataFrame and the
    resulting SAYT corpus.

    Args:
        df: Input DataFrame containing the required columns.
        search_text_col: Column containing searchable text.
        display_text_col: Column containing display text.
        code_col: Column containing codes to normalise.
        incl_code_in_display: Whether to append codes to display text.

    Returns:
        tuple[pd.DataFrame, list[tuple[str, str]]]:
            Updated DataFrame and corpus as `(search_text, display_text)` tuples.
    """
    output_df = df.copy()

    output_df = output_df.rename(columns={code_col: "code"})
    output_df["code"] = output_df["code"].apply(
        lambda x: parse_numerical_code(x).pop(),
    )

    final_display_col = display_text_col

    if incl_code_in_display:
        final_display_col = f"{display_text_col}_with_code"

    if display_text_col not in df.columns:
        raise KeyError(f"Missing required text column: {display_text_col}")
    if code_col not in df.columns:
        raise KeyError(f"Missing required code column: {code_col}")

    output_df = df.copy()
    output_df[final_display_col] = (
        output_df[display_text_col] + ": " + output_df[code_col]
    )

    corpus = list(
        zip(output_df[search_text_col], output_df[final_display_col], strict=False)
    )

    return output_df, corpus


# %%
sayt_corpus_df, sayt_corpus = build_sayt_corpus_from_df(
    df=sayt_df,
    search_text_col="search_text",
    display_text_col="display_text",
    code_col="code",
)

# %%
# Define weights
weights: WeightSpecs = WeightSpecs(
    specs=[
        PrefixWeightSpec(weights={3: 5.0, 6: 4.0}),
        NgramWeightSpec(weights={3: 4.0, 6: 3.0}),
        SemanticWeightSpec(weights={3: 1.0, 6: 3.0}),
    ]
)

# set max_df high to avoid filtering out n-grams in this tiny corpus
suggester = SAYTSuggester(
    sayt_corpus,
    retrievers=[
        PrefixRetrieverSpec(),
        NgramRetrieverSpec(),
        SemanticRetrieverSpec(),
    ],
    weights=weights,
)


# %%
search_query = "Textile machinery wholesale"  # What user types in
expected_title = "Sewing machine wholesale distribution"
characters_typed = search_query[:3]
successful_queries = {}
for character in search_query[3:]:
    characters_typed += character
    query_norm = _normalise(characters_typed)
    print("searching for:", characters_typed)
    print(
        "with score:",
        [
            suggestion.model_dump()
            for suggestion in suggester.suggest_with_scores(
                characters_typed, MAX_SUGGESTIONS
            )
        ],
    )

    returned_titles = suggester.suggest(characters_typed, MAX_SUGGESTIONS)
    titles_list = []
    for title in returned_titles:
        titles_list.append(title[:-7])
    if expected_title in titles_list:
        title_position = titles_list.index(expected_title)
        print(
            f"    Expected title ({expected_title}) found in the list at rank {title_position}."
        )
        successful_queries[characters_typed] = title_position
    else:
        print(f"    Expected title ({expected_title}) NOT in the list.")

    print("easy read:", returned_titles)
    print()
print(successful_queries)
# %%
