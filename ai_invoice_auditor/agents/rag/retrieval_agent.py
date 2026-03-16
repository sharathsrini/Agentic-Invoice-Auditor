import logging

from ai_invoice_auditor.models.state import RAGState
from ai_invoice_auditor.agents.rag.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


def retrieve_node(state: RAGState) -> dict:
    """
    LangGraph node: Retrieves relevant chunks from ChromaDB.
    Queries the shared vectorstore with similarity_search_with_score(query, k=10)
    and returns chunk dicts with text, score, and metadata.
    """
    try:
        query = state.get("query", "")
        logger.info(f"[retrieve_node] Processing query: {query[:80]}")

        vectorstore = get_vectorstore()

        # Handle empty collection gracefully
        try:
            results = vectorstore.similarity_search_with_score(query, k=10)
        except Exception as empty_err:
            logger.warning(f"[retrieve_node] Collection query failed (may be empty): {empty_err}")
            return {
                "retrieved_chunks": [],
                "answer": "No invoice data has been indexed yet.",
                "messages": ["retrieve_node: empty collection"],
            }

        chunks_with_scores = [
            {
                "chunk": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata,
            }
            for doc, score in results
        ]

        logger.info(f"[retrieve_node] Found {len(chunks_with_scores)} chunks")
        return {
            "retrieved_chunks": chunks_with_scores,
            "messages": [f"retrieve_node: found {len(chunks_with_scores)} chunks"],
        }

    except FileNotFoundError as e:
        logger.error(f"[retrieve_node] File not found: {e}")
        return {
            "status": "reject",
            "messages": [f"ERROR [retrieve_node]: file not found -- {e}"],
        }
    except Exception as e:
        logger.error(f"[retrieve_node] Failed: {e}")
        return {
            "retrieved_chunks": [],
            "messages": [f"ERROR [retrieve_node]: {e}"],
        }
