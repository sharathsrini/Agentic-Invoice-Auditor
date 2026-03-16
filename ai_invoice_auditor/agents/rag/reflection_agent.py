import json
import logging
import re

from ai_invoice_auditor.models.state import RAGState
from ai_invoice_auditor.llm import get_llm

logger = logging.getLogger(__name__)

# RAG Triad prompt templates
CONTEXT_RELEVANCE_PROMPT = (
    "On a scale of 0.0 to 1.0, how relevant is the following context to the query?\n"
    "Query: {query}\n"
    "Context: {context}\n"
    'Return ONLY a JSON object: {{"score": <float>}}'
)

GROUNDEDNESS_PROMPT = (
    "On a scale of 0.0 to 1.0, is the following answer fully supported by the context? "
    "Give 1.0 if every claim in the answer appears in the context.\n"
    "Context: {context}\n"
    "Answer: {answer}\n"
    'Return ONLY a JSON object: {{"score": <float>}}'
)

ANSWER_RELEVANCE_PROMPT = (
    "On a scale of 0.0 to 1.0, how relevant is the following answer to the query?\n"
    "Query: {query}\n"
    "Answer: {answer}\n"
    'Return ONLY a JSON object: {{"score": <float>}}'
)


def _parse_score(response_content: str) -> float:
    """Parse a score float from LLM response content.

    Handles JSON objects, markdown code fences, and raw float fallback.
    Returns 0.0 on parse failure.
    """
    content = response_content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if "```" in content:
            content = content.rsplit("```", 1)[0].strip()

    # Try JSON parsing first
    try:
        return float(json.loads(content)["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    # Fallback: find a float pattern in the text
    match = re.search(r"(\d+\.?\d*)", content)
    if match:
        val = float(match.group(1))
        # Clamp to [0.0, 1.0]
        return min(max(val, 0.0), 1.0)

    return 0.0


def reflect_node(state: RAGState) -> dict:
    """
    LangGraph node: Evaluates answer quality using RAG Triad.
    Scores the answer on three axes (context_relevance, groundedness,
    answer_relevance) using LLM-as-judge. Sets passed_reflection and
    increments retry_count for the conditional retry edge.
    """
    try:
        query = state.get("query", "")
        logger.info(f"[reflect_node] Evaluating answer for query: {query[:80]}")

        reranked_chunks = state.get("reranked_chunks", [])
        chunk_texts = [
            c["chunk"] if isinstance(c, dict) else c for c in reranked_chunks
        ]
        context = "\n---\n".join(chunk_texts)
        answer = state.get("answer", "")

        _, chat_model, _ = get_llm()

        # Score each RAG Triad axis
        cr_response = chat_model.invoke(
            CONTEXT_RELEVANCE_PROMPT.format(query=query, context=context)
        )
        cr = _parse_score(cr_response.content)

        gr_response = chat_model.invoke(
            GROUNDEDNESS_PROMPT.format(context=context, answer=answer)
        )
        gr = _parse_score(gr_response.content)

        ar_response = chat_model.invoke(
            ANSWER_RELEVANCE_PROMPT.format(query=query, answer=answer)
        )
        ar = _parse_score(ar_response.content)

        triad_scores = {
            "context_relevance": cr,
            "groundedness": gr,
            "answer_relevance": ar,
        }

        passed = all(v >= 0.7 for v in triad_scores.values())
        triad_scores["passed"] = passed

        retry_count = state.get("retry_count", 0) + 1

        logger.info(f"[reflect_node] scores={triad_scores}, passed={passed}, retry={retry_count}")
        return {
            "triad_scores": triad_scores,
            "passed_reflection": passed,
            "retry_count": retry_count,
            "messages": [f"reflect_node: scores={triad_scores}, passed={passed}"],
        }

    except FileNotFoundError as e:
        logger.error(f"[reflect_node] File not found: {e}")
        return {
            "status": "reject",
            "messages": [f"ERROR [reflect_node]: file not found -- {e}"],
        }
    except Exception as e:
        logger.error(f"[reflect_node] Failed: {e}")
        retry_count = state.get("retry_count", 0) + 1
        return {
            "triad_scores": {
                "context_relevance": 0.0,
                "groundedness": 0.0,
                "answer_relevance": 0.0,
                "passed": False,
            },
            "passed_reflection": False,
            "retry_count": retry_count,
            "messages": [f"ERROR [reflect_node]: {e}"],
        }
