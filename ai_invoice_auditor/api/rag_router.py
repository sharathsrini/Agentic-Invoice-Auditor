"""FastAPI RAG router -- query the RAG pipeline for grounded invoice answers.

Endpoints:
    POST /rag/query -- Submit a natural-language query and get a grounded answer
                       with source chunks and RAG Triad quality scores.
"""

import logging

from fastapi import APIRouter, HTTPException

from ai_invoice_auditor.models.rag import RAGQuery, RAGResponse
from ai_invoice_auditor.observability import get_langfuse_handler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG Q&A"])


def _get_rag_graph():
    """Lazy import to avoid circular imports at module load time."""
    from ai_invoice_auditor.agents.rag.rag_orchestrator import rag_graph
    return rag_graph


@router.post("/query", response_model=RAGResponse)
async def rag_query(request: RAGQuery):
    """Submit a query to the RAG pipeline.

    Invokes the 4-node RAG graph (retrieve -> augment -> generate -> reflect)
    with an optional conditional retry, and returns the grounded answer with
    source chunks and RAG Triad quality scores.

    Request body:
        query: str -- Natural-language question about invoices
        k: int -- Number of chunks to retrieve (default 5)
        filters: dict -- Optional metadata filters

    Returns:
        RAGResponse with answer, source_chunks, triad_scores, and query echo.
    """
    try:
        logger.info("[rag_router] Query received: %s", request.query[:80])

        initial_state = {
            "query": request.query,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "answer": "",
            "triad_scores": {},
            "passed_reflection": False,
            "retry_count": 0,
            "messages": [],
        }

        rag_graph = _get_rag_graph()
        handler = get_langfuse_handler()
        config = {"callbacks": [handler]} if handler else {}
        result = rag_graph.invoke(initial_state, config=config)

        response = RAGResponse(
            answer=result.get("answer", ""),
            source_chunks=result.get("reranked_chunks", []),
            triad_scores=result.get("triad_scores", {}),
            query=request.query,
        )

        logger.info(
            "[rag_router] Query complete: answer_len=%d, passed=%s",
            len(response.answer),
            result.get("passed_reflection", False),
        )
        return response

    except Exception as e:
        logger.error("[rag_router] Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
