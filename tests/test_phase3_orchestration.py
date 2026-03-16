"""Phase 3 orchestration verification suite.

Structural and import-level tests validating all Phase 3 features:
  1. LangGraph pipeline graph compilation (7 nodes + conditional edges)
  2. MCP server tool registration (10 tools)
  3. MCP client call_tool fallback path
  4. FastAPI invoice router (4 endpoints)
  5. main.py dual-router mounting (erp + invoice + health)
  6. Report node HTML generation (6 sections)
  7. Validator nodes MCP client wiring

No live LLM calls or external API access -- all tests are deterministic.
"""

import inspect
import os
from pathlib import Path

import pytest

# Load .env for GEMINI_API_KEY (needed by some transitive imports)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Manual fallback: read .env file if python-dotenv not installed
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


# ---- Test 1: Pipeline graph compiles with 7 nodes ----

def test_pipeline_graph_compiles_with_7_nodes():
    """Verify LangGraph pipeline_graph has 7 user-defined nodes and conditional edges."""
    from ai_invoice_auditor.agents.orchestrator import pipeline_graph

    assert pipeline_graph is not None, "pipeline_graph should not be None"

    # Access the graph structure
    graph = pipeline_graph.get_graph()
    node_ids = set(graph.nodes.keys())

    # Expected 7 user-defined nodes
    expected_nodes = {
        "monitor_node",
        "extract_node",
        "translate_node",
        "data_validate_node",
        "biz_validate_node",
        "report_node",
        "index_node",
    }

    # Filter out LangGraph meta-nodes (__start__, __end__)
    user_nodes = node_ids - {"__start__", "__end__"}
    assert expected_nodes == user_nodes, (
        f"Expected nodes {expected_nodes}, got {user_nodes}"
    )

    # Verify conditional edge from biz_validate_node to report_node exists
    edges = graph.edges
    # edges is a set of (source, target) tuples (or Edge objects)
    biz_to_report = False
    for edge in edges:
        # Handle both tuple and Edge object forms
        if hasattr(edge, "source"):
            src, tgt = edge.source, edge.target
        else:
            src, tgt = edge[0], edge[1]
        if src == "biz_validate_node" and tgt == "report_node":
            biz_to_report = True
            break

    assert biz_to_report, (
        "Expected conditional edge from biz_validate_node to report_node"
    )


# ---- Test 2: MCP server registers 10 tools ----

def test_mcp_server_registers_10_tools():
    """Verify FastMCP server has exactly 10 registered tools."""
    from ai_invoice_auditor.mcp_server import mcp

    # FastMCP 2.x: access tools via internal _tool_manager
    if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
        tools = mcp._tool_manager._tools
        tool_names = set(tools.keys())
    else:
        # Fallback: try other known access patterns
        raise RuntimeError(
            "Cannot access FastMCP tool registry. "
            "Tried mcp._tool_manager._tools"
        )

    expected_tools = {
        "invoice_watcher",
        "data_harvester",
        "lang_bridge",
        "completeness_checker",
        "business_validator",
        "insight_reporter",
        "vector_indexer",
        "semantic_retriever",
        "chunk_ranker",
        "response_synthesizer",
    }

    assert len(tool_names) == 10, (
        f"Expected 10 tools, got {len(tool_names)}: {tool_names}"
    )
    assert expected_tools == tool_names, (
        f"Tool mismatch. Missing: {expected_tools - tool_names}, "
        f"Extra: {tool_names - expected_tools}"
    )


# ---- Test 3: MCP client call_tool fallback ----

def test_mcp_client_call_tool_fallback():
    """Verify call_tool falls back to fallback_fn when MCP call fails."""
    from ai_invoice_auditor.mcp_client import call_tool

    def my_fallback(**kwargs):
        return {"fallback": True, **kwargs}

    result = call_tool(
        "nonexistent_tool",
        {"key": "value"},
        fallback_fn=my_fallback,
    )

    assert result == {"fallback": True, "key": "value"}, (
        f"Fallback result mismatch: {result}"
    )


# ---- Test 4: FastAPI invoice router has 4 endpoints ----

def test_fastapi_invoice_router_has_4_endpoints():
    """Verify invoice_router has exactly 4 APIRoute endpoints."""
    from ai_invoice_auditor.api.invoice_router import router
    from fastapi.routing import APIRoute

    api_routes = [r for r in router.routes if isinstance(r, APIRoute)]
    assert len(api_routes) == 4, (
        f"Expected 4 API routes, got {len(api_routes)}: "
        f"{[(r.path, r.methods) for r in api_routes]}"
    )

    # Verify specific paths and methods
    route_map = {r.path: r.methods for r in api_routes}

    expected = {
        "/invoice/process": {"POST"},
        "/invoice/{invoice_no}/resume": {"POST"},
        "/invoice/{invoice_no}/report": {"GET"},
        "/invoice/{invoice_no}": {"GET"},
    }

    for path, methods in expected.items():
        assert path in route_map, f"Missing route: {path}"
        assert methods == route_map[path], (
            f"Route {path}: expected methods {methods}, got {route_map[path]}"
        )


# ---- Test 5: main.py app mounts both routers ----

def test_main_app_mounts_both_routers():
    """Verify FastAPI app includes erp, invoice, and health routes."""
    from ai_invoice_auditor.main import app
    from fastapi.routing import APIRoute

    api_routes = [r for r in app.routes if isinstance(r, APIRoute)]
    all_paths = {r.path for r in api_routes}

    # Check for /erp prefixed routes
    erp_paths = {p for p in all_paths if p.startswith("/erp")}
    assert len(erp_paths) > 0, f"No /erp routes found. All paths: {all_paths}"

    # Check for /invoice prefixed routes
    invoice_paths = {p for p in all_paths if p.startswith("/invoice")}
    assert len(invoice_paths) > 0, (
        f"No /invoice routes found. All paths: {all_paths}"
    )

    # Check for /health endpoint
    assert "/health" in all_paths, (
        f"/health not found. All paths: {all_paths}"
    )


# ---- Test 6: report_node generates HTML with 6 sections ----

def test_report_node_generates_html_with_6_sections():
    """Verify report_node produces HTML containing all 6 template sections."""
    from ai_invoice_auditor.agents.reporter_agent import report_node

    state = {
        "file_path": "test.pdf",
        "meta": {},
        "raw_text": "test",
        "extracted": {
            "invoice_no": "TEST-001",
            "vendor_name": "Acme",
            "vendor_id": "V001",
            "invoice_date": "2026-01-01",
            "po_number": "PO-100",
            "currency": "USD",
            "total_amount": 1000.00,
        },
        "translated": {"is_english": True},
        "validation_result": {
            "missing_fields": [],
            "all_missing": [],
            "discrepancies": [],
            "status": "auto_approve",
        },
        "report_path": None,
        "messages": ["test message"],
        "status": "auto_approve",
        "next": "",
    }

    result = report_node(state)

    assert "report_path" in result, "report_node result should contain report_path"
    report_path = result["report_path"]
    assert report_path and len(report_path) > 0, "report_path should be non-empty"

    # Read the generated HTML
    html_content = Path(report_path).read_text(encoding="utf-8")

    # Assert all 6 section IDs are present
    expected_sections = [
        'id="invoice-summary"',
        'id="translation-details"',
        'id="data-completeness"',
        'id="business-validation"',
        'id="recommendation"',
        'id="audit-trail"',
    ]

    for section_id in expected_sections:
        assert section_id in html_content, (
            f"Missing section {section_id} in generated HTML"
        )

    # Clean up generated report file
    try:
        Path(report_path).unlink()
    except OSError:
        pass


# ---- Test 7: Validator nodes use MCP client wrappers ----

def test_validator_nodes_use_mcp_client_wrappers():
    """Verify data_validate_node and business_validate_node use MCP client wrappers."""
    from ai_invoice_auditor.agents.data_validator_agent import data_validate_node
    from ai_invoice_auditor.agents.business_validator_agent import business_validate_node
    import ai_invoice_auditor.agents.data_validator_agent as dv_module
    import ai_invoice_auditor.agents.business_validator_agent as bv_module

    # Check source code of data_validate_node for MCP client wrapper usage
    dv_source = inspect.getsource(data_validate_node)
    assert "call_completeness_checker" in dv_source, (
        "data_validate_node should call call_completeness_checker (MCP client wrapper)"
    )

    # Check source code of business_validate_node for MCP client wrapper usage
    bv_source = inspect.getsource(business_validate_node)
    assert "call_business_validator" in bv_source, (
        "business_validate_node should call call_business_validator (MCP client wrapper)"
    )

    # Verify module-level imports
    assert hasattr(dv_module, "call_completeness_checker"), (
        "data_validator_agent module should have call_completeness_checker in namespace"
    )
    assert hasattr(bv_module, "call_business_validator"), (
        "business_validator_agent module should have call_business_validator in namespace"
    )
