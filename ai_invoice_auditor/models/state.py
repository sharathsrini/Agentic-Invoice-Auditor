"""LangGraph state TypedDicts for the invoice and RAG pipelines."""

from operator import add
from typing import Annotated, Optional, TypedDict


class InvoiceState(TypedDict):
    """State container for the invoice processing pipeline."""

    file_path: str
    meta: dict
    raw_text: Optional[str]
    extracted: Optional[dict]
    translated: Optional[dict]
    validation_result: Optional[dict]
    report_path: Optional[str]
    messages: Annotated[list, add]
    status: str
    next: str


class RAGState(TypedDict):
    """State container for the RAG query pipeline."""

    query: str
    retrieved_chunks: list[str]
    reranked_chunks: list[dict]
    answer: str
    triad_scores: dict
    passed_reflection: bool
    retry_count: int
    messages: Annotated[list, add]
