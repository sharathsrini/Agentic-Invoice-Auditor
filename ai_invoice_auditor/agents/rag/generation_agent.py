import logging

from ai_invoice_auditor.models.state import RAGState
from ai_invoice_auditor.llm import get_llm

logger = logging.getLogger(__name__)


def generate_node(state: RAGState) -> dict:
    """
    LangGraph node: Generates grounded answer from top chunks.
    Uses the active chat model with a grounding prompt that forces the LLM
    to answer strictly from context or respond with "I don't have that information."
    """
    try:
        query = state.get("query", "")
        logger.info(f"[generate_node] Processing query: {query[:80]}")

        reranked_chunks = state.get("reranked_chunks", [])
        # Extract chunk text (handle both dict and plain string formats)
        chunk_texts = [
            c["chunk"] if isinstance(c, dict) else c for c in reranked_chunks
        ]

        if not chunk_texts:
            return {
                "answer": "I don't have that information.",
                "messages": ["generate_node: no context chunks"],
            }

        _, chat_model, _ = get_llm()

        joined_context = "\n---\n".join(chunk_texts)

        prompt = (
            "You are an invoice audit assistant. Answer the question using ONLY the context below.\n"
            'If the answer is not in the context, respond with exactly: "I don\'t have that information."\n'
            "Do not make up information or use knowledge outside the provided context.\n\n"
            f"Context:\n{joined_context}\n\n"
            f"Question: {query}"
        )

        response = chat_model.invoke(prompt)
        answer = response.content.strip()

        logger.info(f"[generate_node] Generated answer ({len(answer)} chars)")
        return {
            "answer": answer,
            "messages": [f"generate_node: generated answer ({len(answer)} chars)"],
        }

    except FileNotFoundError as e:
        logger.error(f"[generate_node] File not found: {e}")
        return {
            "status": "reject",
            "messages": [f"ERROR [generate_node]: file not found -- {e}"],
        }
    except Exception as e:
        logger.error(f"[generate_node] Failed: {e}")
        return {
            "answer": "I don't have that information.",
            "messages": [f"ERROR [generate_node]: {e}"],
        }
