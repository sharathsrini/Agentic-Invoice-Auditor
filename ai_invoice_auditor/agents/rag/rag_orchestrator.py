"""LangGraph RAG query graph with conditional retry edge.

Wires four RAG agent nodes into a sequential graph with reflection-based
quality control:

    retrieve_node -> augment_node -> generate_node -> reflect_node
                                          ^                |
                                          |   (retry)      |
                                          +----------------+

After reflection, the conditional edge `route_after_reflect` checks:
  - If all RAG Triad scores pass (>= 0.7) -> END (return answer)
  - If scores fail and retry_count < 2    -> retry (re-run generate_node)
  - If scores fail and retry_count >= 2   -> END (return best-effort answer)

This allows exactly 1 retry: reflect_node increments retry_count on each call.
First reflection sets retry_count=1; if it fails, we retry generate_node then
reflect again setting retry_count=2, at which point we return regardless.

No checkpointer -- stateless Q&A, no persistence needed.
"""

import logging

from langgraph.graph import StateGraph, START, END

from ai_invoice_auditor.models.state import RAGState
from ai_invoice_auditor.agents.rag.retrieval_agent import retrieve_node
from ai_invoice_auditor.agents.rag.augmentation_agent import augment_node
from ai_invoice_auditor.agents.rag.generation_agent import generate_node
from ai_invoice_auditor.agents.rag.reflection_agent import reflect_node

logger = logging.getLogger(__name__)


def route_after_reflect(state: RAGState) -> str:
    """Route after reflection: end on pass/max-retry, retry on fail.

    Routing logic:
      - passed_reflection == True  -> "end" (quality is acceptable)
      - retry_count >= 2           -> "end" (max 1 retry exhausted)
      - otherwise                  -> "retry" (re-run generation)
    """
    if state.get("passed_reflection", False):
        logger.info("[route_after_reflect] Scores passed -> END")
        return "end"
    if state.get("retry_count", 0) >= 2:
        logger.warning(
            "[route_after_reflect] Max retries reached (retry_count=%d) -> END",
            state.get("retry_count", 0),
        )
        return "end"
    logger.info(
        "[route_after_reflect] Scores failed, retry_count=%d -> RETRY",
        state.get("retry_count", 0),
    )
    return "retry"


# --- Build the RAG graph ---

builder = StateGraph(RAGState)

builder.add_node("retrieve_node", retrieve_node)
builder.add_node("augment_node", augment_node)
builder.add_node("generate_node", generate_node)
builder.add_node("reflect_node", reflect_node)

builder.add_edge(START, "retrieve_node")
builder.add_edge("retrieve_node", "augment_node")
builder.add_edge("augment_node", "generate_node")
builder.add_edge("generate_node", "reflect_node")
builder.add_conditional_edges(
    "reflect_node",
    route_after_reflect,
    {"retry": "generate_node", "end": END},
)

rag_graph = builder.compile()

logger.info(
    "RAG graph compiled: nodes=%s",
    list(rag_graph.nodes.keys()),
)
