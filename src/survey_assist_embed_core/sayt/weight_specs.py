"""Weight specification for retriever combination."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


class RetrieverWeightSpec(Protocol):
    """Weighting contract for individual weight configurations.

    Defines the interface that the SAYT orchestrator expects when working
    with weight specs. Custom weight implementations should conform to this.
    """

    @property
    def retriever_name(self) -> str:
        """Name of the retriever this weight applies to."""
        ...

    @property
    def weights(self) -> int | float | dict[int, float]:
        """Raw weight configuration (fixed number or query-length-specific)."""
        ...

    def get_weight(self, query_length: int) -> float:
        """Get the weight for a given query length.

        Args:
            query_length: The normalized query length in characters.

        Returns:
            The weight value for this retriever at the given query length.
        """
        ...


def _validate_retriever_weight(weight: float) -> None:
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("retriever weight must be a finite value > 0")


@dataclass(frozen=True, slots=True)
class WeightConfig:
    """Weight configuration for a retriever.

    Weights can be either:
    - A number (int or float): fixed weight used for all query lengths
    - A dict[int, float]: query-length-specific weights using nearest lower bound
    """

    weights: int | float | dict[int, float] = 1.0
    retriever_name: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate weights."""
        if isinstance(self.weights, (int, float)):
            _validate_retriever_weight(float(self.weights))
        else:
            for length, weight in self.weights.items():
                if not isinstance(length, int) or length <= 0:
                    raise ValueError(f"Query length must be positive int, got {length}")
                _validate_retriever_weight(weight)

    def get_weight(self, query_length: int | None = None) -> float:
        """Get weight for this retriever at a query length.

        Args:
            query_length: The normalized query length in characters.

        Returns:
            The weight for this retriever at the given query length.
            If weights is a number, returns that value.
            If weights is a dict, uses nearest lower bound lookup, or 0.0 if not found.
        """
        if isinstance(self.weights, (int, float)):
            return float(self.weights)

        # Dictionary lookup with nearest lower bound
        if query_length is None:
            raise ValueError("query_length must be provided for dict weights")

        valid_lengths = [length for length in self.weights if length <= query_length]
        if not valid_lengths:
            return 0.0
        best_length = max(valid_lengths)
        return self.weights[best_length]


@dataclass(frozen=True, slots=True)
class PrefixWeightSpec(WeightConfig):
    """Weight configuration for prefix retriever."""

    retriever_name: str = field(init=False, default="prefix")
    weights: int | float | dict[int, float] = field(default=1.0)


@dataclass(frozen=True, slots=True)
class NgramWeightSpec(WeightConfig):
    """Weight configuration for n-gram retriever."""

    retriever_name: str = field(init=False, default="ngram")
    weights: int | float | dict[int, float] = field(default=1.0)


@dataclass(frozen=True, slots=True)
class SemanticWeightSpec(WeightConfig):
    """Weight configuration for semantic retriever."""

    retriever_name: str = field(init=False, default="semantic")
    weights: int | float | dict[int, float] = field(default=1.0)


@dataclass(frozen=True, slots=True)
class WeightSpecs:
    """Collection of weight specs as an ordered sequence."""

    specs: Sequence[WeightConfig] = field(default_factory=list)

    def get_weight_spec(self, retriever_name: str) -> WeightConfig | None:
        """Get weight spec for a specific retriever by name.

        Args:
            retriever_name: Name of the retriever (e.g., "prefix", "ngram", "semantic").

        Returns:
            The WeightConfig for this retriever, or None if not found.
        """
        for spec in self.specs:
            if spec.retriever_name == retriever_name:
                return spec
        return None

    def get_weight(self, retriever_name: str, query_length: int) -> float:
        """Get weight for a specific retriever at a query length.

        Args:
            retriever_name: Name of the retriever.
            query_length: The normalized query length in characters.

        Returns:
            The weight for the retriever, or 0.0 if retriever not found.
        """
        spec = self.get_weight_spec(retriever_name)
        if spec is None:
            return 0.0
        return spec.get_weight(query_length)


def default_weight_specs() -> WeightSpecs:
    """Return the standard runtime weight specs used by SAYT.

    Returns:
        WeightSpecs container with default prefix, n-gram, and semantic specs.
    """
    return WeightSpecs(
        specs=[
            PrefixWeightSpec(),
            NgramWeightSpec(),
            SemanticWeightSpec(),
        ]
    )
