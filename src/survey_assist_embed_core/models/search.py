"""Retrieval response models shared by embedding and vector-search services."""

from pydantic import BaseModel


class SearchIndexItem(BaseModel):
    """Represent one ranked vector-search result."""

    distance: float
    title: str
    code: str


class SearchIndexResponse(BaseModel):
    """Represent a vector-search response payload."""

    results: list[SearchIndexItem]
