"""Build and reload a persisted SAYT artifact with the built-in retrievers."""

# pylint: disable=duplicate-code

# %%
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    NgramWeightSpec,
    PrefixRetrieverSpec,
    PrefixWeightSpec,
    RetrieverSpec,
    SAYTBuilder,
    SAYTSuggester,
    SemanticRetrieverSpec,
    SemanticWeightSpec,
    WeightSpecs,
)

# %%
############# toy example to verify SAYT artifact build/load works #############
small_corpus = [
    ("Car wash", "Car Wash"),
    ("Car wash", "CAR WASH (duplicate)"),
    ("Car waxing", "Car Waxing"),
    ("Waxing car", "Car Waxing"),
    ("Carpentry services", "Carpentry services"),
    ("Dog grooming", "Dog grooming"),
    ("Cat grooming", "Cat grooming"),
    ("USed car sales", "Used car sales"),
    ("Car rental", "Car rental"),
    ("Car repair", "Car repair"),
    ("Car servicing", "Car servicing"),
]

retrievers: list[RetrieverSpec] = [
    PrefixRetrieverSpec(),
    NgramRetrieverSpec(max_df=0.8),
    SemanticRetrieverSpec(),
]

weights: WeightSpecs = WeightSpecs(
    specs=[
        PrefixWeightSpec(),
        NgramWeightSpec(weights=2),
        SemanticWeightSpec(weights={4: 2.0, 6: 3.0}),
    ]
)

# Keep the temporary directory alive across notebook cells.
# pylint: disable-next=consider-using-with
temp_dir = TemporaryDirectory(prefix="sayt_artifact_demo_")
artifact_dir = Path(temp_dir.name) / "car_services_sayt"
print("artifact will be written to:", artifact_dir)


# %%
# Semantic artifact builds may take longer the first time if the model cache
# needs to be created locally.
artifact_path = SAYTBuilder(
    small_corpus,
    retrievers=retrievers,
    min_chars=3,
    max_suggestions=5,
    include_tied_suggestions=False,
    weights=weights,
).build_artifact(artifact_dir, overwrite=True)

print("artifact saved to:", artifact_path)
print("artifact files:")
for path in sorted(artifact_path.rglob("*")):
    if path.is_file():
        print("-", path.relative_to(artifact_path))


# %%
manifest = json.loads((artifact_path / "manifest.json").read_text(encoding="utf-8"))
print(json.dumps(manifest, indent=2))


# %%
live_suggester = SAYTSuggester(
    small_corpus,
    retrievers=retrievers,
    min_chars=3,
    max_suggestions=5,
    include_tied_suggestions=False,
    weights=weights,
)
loaded_suggester = SAYTSuggester.from_artifact(artifact_path)

for query in ["car", "cars", "waxi", "grom", "wash", "duplicate", "auto"]:
    live_suggestions = live_suggester.suggest(query, 5)
    loaded_suggestions = loaded_suggester.suggest(query, 5)

    print("searching for:", query)
    print("live", "->", live_suggestions)
    print("loaded", "->", loaded_suggestions)
    print(
        "loaded_scores",
        "->",
        [
            suggestion.model_dump()
            for suggestion in loaded_suggester.suggest_with_scores(query, 5)
        ],
    )
    if live_suggestions != loaded_suggestions:
        raise RuntimeError("Loaded suggester results did not match live build")
    print()


# %%
query = "car"
print("strict limit (configured in the artifact):", loaded_suggester.suggest(query, 2))
print(
    "include cutoff ties (may exceed 2 suggestions):",
    loaded_suggester.suggest(query, 2, include_tied_suggestions=True),
)


# %%
# Build a separate prefix-only artifact so "car" produces exact cutoff ties.
prefix_artifact_path = SAYTBuilder(
    small_corpus,
    retrievers=[PrefixRetrieverSpec()],
    weights=WeightSpecs(specs=[PrefixWeightSpec()]),
    min_chars=3,
    max_suggestions=2,
    include_tied_suggestions=False,
).build_artifact(Path(temp_dir.name) / "prefix_cutoff_ties")
prefix_suggester = SAYTSuggester.from_artifact(prefix_artifact_path)

prefix_strict_results = prefix_suggester.suggest_with_scores("car")
prefix_tied_results = prefix_suggester.suggest_with_scores(
    "car", include_tied_suggestions=True
)
print(
    "prefix artifact, strict limit:",
    [suggestion.model_dump() for suggestion in prefix_strict_results],
)
print(
    "prefix artifact, including cutoff ties:",
    [suggestion.model_dump() for suggestion in prefix_tied_results],
)

# %%
# Persist ties as the default, then override that setting for one query.
prefix_ties_artifact_path = SAYTBuilder(
    small_corpus,
    retrievers=[PrefixRetrieverSpec()],
    weights=WeightSpecs(specs=[PrefixWeightSpec()]),
    min_chars=3,
    max_suggestions=2,
    include_tied_suggestions=True,
).build_artifact(Path(temp_dir.name) / "prefix_cutoff_ties_enabled")
prefix_ties_suggester = SAYTSuggester.from_artifact(prefix_ties_artifact_path)

prefix_ties_default_results = prefix_ties_suggester.suggest_with_scores("car")
prefix_ties_strict_results = prefix_ties_suggester.suggest_with_scores(
    "car", include_tied_suggestions=False
)
print(
    "ties-enabled artifact, including cutoff ties by default:",
    [suggestion.model_dump() for suggestion in prefix_ties_default_results],
)
print(
    "ties-enabled artifact, strict limit override:",
    [suggestion.model_dump() for suggestion in prefix_ties_strict_results],
)
# %%
# Run `temp_dir.cleanup()` when you are finished exploring the saved files.
print("artifact ready for inspection:", artifact_path)


# %%
temp_dir.cleanup()
# %%
