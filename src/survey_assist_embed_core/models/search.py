"""Retrieval response models shared by embedding and vector-search services."""

from pydantic import BaseModel


class SearchIndexItem(BaseModel):
    """One ranked item returned from a vector search.

    Attributes:
        distance: Distance derived from the backend similarity score.
        title: Human-readable title for the matched classification item.
        code: Classification code associated with the title.
    """

    distance: float
    title: str
    code: str


class SearchIndexResponse(BaseModel):
    """Vector-search response payload.

    Attributes:
        results: Ranked search results in response order.
    """

    results: list[SearchIndexItem]
