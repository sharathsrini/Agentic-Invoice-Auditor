"""RAG (Retrieval-Augmented Generation) data models."""

from typing import Optional

from pydantic import BaseModel


class RAGQuery(BaseModel):
    """A query to the RAG system."""

    query: str
    k: int = 5
    filters: Optional[dict] = None


class RAGResponse(BaseModel):
    """Response from the RAG system."""

    answer: str
    source_chunks: list[dict] = []
    triad_scores: dict = {}
    query: str
