"""LangGraph orchestrator -- 7-node pipeline graph with MemorySaver and conditional edges.

Compiles the full invoice processing pipeline:
  START -> monitor_node -> extract_node -> translate_node -> data_validate_node
        -> biz_validate_node --(conditional)--> report_node -> index_node -> END

The conditional edge after biz_validate_node routes based on state["status"].
All status values route to report_node for HTML report generation.
The edit re-route is handled inside biz_validate_node via Command(goto='data_validate_node'),
which LangGraph processes BEFORE evaluating conditional edges.

Usage:
    from ai_invoice_auditor.agents.orchestrator import pipeline_graph
    result = pipeline_graph.invoke(state, config={"configurable": {"thread_id": "invoice-123"}})
"""

import logging

from langgraph.graph import StateGraph, START, END

try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver

from ai_invoice_auditor.models.state import InvoiceState
from ai_invoice_auditor.agents.monitor_agent import monitor_node
from ai_invoice_auditor.agents.extractor_agent import extract_node
from ai_invoice_auditor.agents.translator_agent import translate_node
from ai_invoice_auditor.agents.data_validator_agent import data_validate_node
from ai_invoice_auditor.agents.business_validator_agent import business_validate_node
from ai_invoice_auditor.agents.reporter_agent import report_node
from ai_invoice_auditor.agents.rag.indexing_agent import index_node

logger = logging.getLogger(__name__)


def route_after_biz_validate(state: InvoiceState) -> str:
    """Route after business validation based on status.

    All status values route to report_node for HTML report generation.
    The edit re-route is handled inside biz_validate_node via Command(goto='data_validate_node')
    which overrides the graph edge -- LangGraph processes Command(goto) before checking edges.
    """
    status = state.get("status", "")
    if status in ("reject", "flag", "manual_review", "auto_approve", ""):
        return "report_node"
    return "report_node"  # default: always generate report


def build_pipeline_graph():
    """Build and compile the 7-node invoice processing pipeline graph.

    Returns:
        Compiled LangGraph StateGraph with MemorySaver checkpointer.
    """
    builder = StateGraph(InvoiceState)

    # Add all 7 nodes
    builder.add_node("monitor_node", monitor_node)
    builder.add_node("extract_node", extract_node)
    builder.add_node("translate_node", translate_node)
    builder.add_node("data_validate_node", data_validate_node)
    builder.add_node("biz_validate_node", business_validate_node)
    builder.add_node("report_node", report_node)
    builder.add_node("index_node", index_node)

    # Wire linear edges
    builder.add_edge(START, "monitor_node")
    builder.add_edge("monitor_node", "extract_node")
    builder.add_edge("extract_node", "translate_node")
    builder.add_edge("translate_node", "data_validate_node")
    builder.add_edge("data_validate_node", "biz_validate_node")

    # Conditional edge after biz_validate_node (ORCH-02)
    builder.add_conditional_edges(
        "biz_validate_node",
        route_after_biz_validate,
        {"report_node": "report_node"},
    )

    builder.add_edge("report_node", "index_node")
    builder.add_edge("index_node", END)

    # Compile with MemorySaver for HITL state persistence
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    logger.info(
        "Pipeline graph compiled with 7 nodes, conditional edges, and MemorySaver checkpointer"
    )
    return graph


# Module-level compiled graph instance
pipeline_graph = build_pipeline_graph()
