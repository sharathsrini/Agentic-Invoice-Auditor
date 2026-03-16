import json
import logging

from ai_invoice_auditor.models.state import RAGState
from ai_invoice_auditor.llm import get_llm

logger = logging.getLogger(__name__)


def augment_node(state: RAGState) -> dict:
    """
    LangGraph node: Reranks retrieved chunks using LLM.
    Takes top-10 retrieved chunks, asks the LLM to rank by relevance,
    and returns the top-3 with relevance scores.
    """
    try:
        retrieved_chunks = state.get("retrieved_chunks", [])
        if not retrieved_chunks:
            return {
                "reranked_chunks": [],
                "messages": ["augment_node: no chunks to rerank"],
            }

        query = state.get("query", "")
        logger.info(f"[augment_node] Reranking {len(retrieved_chunks)} chunks for query: {query[:80]}")

        # Extract chunk texts (handle both dict and plain string formats)
        chunk_texts = [
            c["chunk"] if isinstance(c, dict) else c for c in retrieved_chunks
        ]

        _, chat_model, _ = get_llm()

        # Build numbered list of chunks for the LLM prompt
        numbered_chunks = "\n".join(
            f"{i + 1}. {text[:500]}" for i, text in enumerate(chunk_texts)
        )

        prompt = (
            f'You are a relevance ranking assistant.\n'
            f'Query: "{query}"\n\n'
            f'Rank the following chunks by relevance to the query.\n'
            f'Return a JSON array of objects for the top 3 most relevant, in order of relevance.\n'
            f'Each object must have: "index" (1-based int) and "score" (float 0.0-1.0).\n'
            f'Return ONLY the JSON array, no other text.\n\n'
            f'Chunks:\n{numbered_chunks}'
        )

        response = chat_model.invoke(prompt)
        content = response.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            # Remove opening fence (e.g., ```json\n)
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            # Remove closing fence
            if "```" in content:
                content = content.rsplit("```", 1)[0].strip()

        try:
            rankings = json.loads(content)
            top3 = []
            for entry in rankings[:3]:
                idx = entry["index"] - 1  # Convert 1-based to 0-based
                if 0 <= idx < len(chunk_texts):
                    top3.append({
                        "chunk": chunk_texts[idx],
                        "score": float(entry["score"]),
                    })
            logger.info(f"[augment_node] LLM reranked to top {len(top3)}")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as parse_err:
            logger.warning(f"[augment_node] JSON parse fallback: {parse_err}")
            # Fallback: return first 3 chunks with score 0.5
            top3 = [
                {"chunk": text, "score": 0.5}
                for text in chunk_texts[:3]
            ]

        return {
            "reranked_chunks": top3,
            "messages": [f"augment_node: reranked to top {len(top3)}"],
        }

    except FileNotFoundError as e:
        logger.error(f"[augment_node] File not found: {e}")
        return {
            "status": "reject",
            "messages": [f"ERROR [augment_node]: file not found -- {e}"],
        }
    except Exception as e:
        logger.error(f"[augment_node] Failed: {e}")
        return {
            "reranked_chunks": [],
            "messages": [f"ERROR [augment_node]: {e}"],
        }
