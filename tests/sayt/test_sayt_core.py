"""Unit tests for SAYT core ranking and cutoff handling."""

import pytest

from survey_assist_embed_core.sayt.core import Suggestion, take_with_ties


def test_take_with_ties_includes_cutoff_ties_by_default():
    """Preserve cutoff ties when callers omit the flag."""
    items = [
        Suggestion(display_text="Alpha", score=1.0),
        Suggestion(display_text="Beta", score=1.0),
        Suggestion(display_text="Gamma", score=0.5),
    ]

    assert take_with_ties(items, limit=1) == items[:2]


@pytest.mark.parametrize(
    ("include_tied_suggestions", "expected_display_texts"),
    [
        (False, ["Top", "Zulu"]),
        (True, ["Top", "Zulu", "alpha", "Beta"]),
    ],
)
def test_take_with_ties_applies_limit_after_ranking_and_deduplication(
    include_tied_suggestions, expected_display_texts
):
    """Apply cutoff policy after score, count and case-insensitive ranking."""
    items = [
        Suggestion(display_text="Beta", score=0.8),
        Suggestion(display_text="Top", score=0.9),
        Suggestion(display_text="Zulu", score=0.8),
        Suggestion(display_text="Top", score=1.0),
        Suggestion(display_text="alpha", score=0.8),
        Suggestion(display_text="Lower", score=0.5),
    ]

    results = take_with_ties(
        items,
        limit=2,
        include_tied_suggestions=include_tied_suggestions,
        display_text_value_counts={"Zulu": 3},
    )

    assert [item.display_text for item in results] == expected_display_texts
    assert results[0].score == 1.0


@pytest.mark.parametrize("include_tied_suggestions", [False, True])
@pytest.mark.parametrize(
    ("display_texts", "limit", "expected_display_texts"),
    [
        ([], 2, []),
        (["Alpha", "Beta"], 0, []),
        (["Alpha", "Beta"], -1, []),
        (["Alpha"], 1, ["Alpha"]),
        (["Alpha", "Alpha", "Beta"], 2, ["Alpha", "Beta"]),
        (["Alpha", "Alpha", "Beta"], 5, ["Alpha", "Beta"]),
    ],
)
def test_take_with_ties_handles_limit_boundaries(
    include_tied_suggestions, display_texts, limit, expected_display_texts
):
    """Handle empty inputs and limits at or outside the unique result count."""
    items = [Suggestion(display_text=text, score=1.0) for text in display_texts]

    results = take_with_ties(
        items, limit=limit, include_tied_suggestions=include_tied_suggestions
    )

    assert [item.display_text for item in results] == expected_display_texts


@pytest.mark.parametrize("include_tied_suggestions", [False, True])
def test_take_with_ties_does_not_extend_cutoff_without_a_tie(include_tied_suggestions):
    """Stop at the requested limit when the next score is lower."""
    items = [
        Suggestion(display_text="Lower", score=0.5),
        Suggestion(display_text="Top", score=1.0),
        Suggestion(display_text="Middle", score=0.8),
    ]

    results = take_with_ties(
        items, limit=2, include_tied_suggestions=include_tied_suggestions
    )

    assert results == [items[1], items[2]]
