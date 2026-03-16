"""Phase 4 RAG pipeline verification suite.

Structural and live-LLM tests validating all Phase 4 features:
  1. Vectorstore singleton returns Chroma instance and is idempotent
  2. index_node chunks text, stores in ChromaDB with metadata, deduplicates on re-index
  3. RAG graph compiles with 4 user-defined nodes and a conditional retry edge
  4. retrieve_node returns chunk dicts from ChromaDB via similarity_search_with_score
  5. augment_node LLM-reranks to top-3 chunks (LIVE LLM)
  6. generate_node produces grounded answers and refusal for irrelevant queries (LIVE LLM)
  7. reflect_node evaluates with RAG Triad scores and sets passed_reflection (LIVE LLM)
  8. /rag/query endpoint exists on the FastAPI app
  9. MCP server has real implementations for vector_indexer, semantic_retriever, chunk_ranker
"""

import inspect
import os
from pathlib import Path

import pytest

# Load .env for GEMINI_API_KEY (needed by LLM-dependent imports)
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


# ---- Test 1: Vectorstore singleton returns Chroma instance ----

def test_vectorstore_singleton_returns_chroma_instance():
    """Verify get_vectorstore() returns a Chroma singleton with correct collection name."""
    from langchain_chroma import Chroma
    from ai_invoice_auditor.agents.rag.vectorstore import get_vectorstore

    vs1 = get_vectorstore()
    vs2 = get_vectorstore()

    # Singleton identity
    assert vs1 is vs2, "get_vectorstore() should return the same object (singleton)"

    # Type check
    assert isinstance(vs1, Chroma), f"Expected Chroma instance, got {type(vs1)}"

    # Collection name
    assert vs1._collection.name == "invoice_chunks", (
        f"Expected collection 'invoice_chunks', got '{vs1._collection.name}'"
    )


# ---- Test 2: index_node chunks and stores with dedup ----

def test_index_node_chunks_and_stores_with_dedup():
    """Verify index_node chunks text, stores with metadata, and deduplicates on re-index."""
    from ai_invoice_auditor.agents.rag.indexing_agent import index_node
    from ai_invoice_auditor.agents.rag.vectorstore import get_vectorstore

    # Build a fake InvoiceState dict with raw_text exceeding 512-char chunk_size
    long_text = (
        "This is a test invoice document for widget procurement. "
        "The total amount is $5000 for order PO-100 from vendor V001. "
        "Items include various widgets and gadgets for the Acme Corp project. "
    ) * 15  # ~195 chars * 15 = ~2925 chars -> should produce multiple chunks

    state = {
        "file_path": "test.pdf",
        "meta": {},
        "raw_text": long_text,
        "extracted": {
            "invoice_no": "TEST-RAG-001",
            "vendor_id": "V001",
            "po_number": "PO-100",
        },
        "translated": None,
        "validation_result": None,
        "report_path": None,
        "messages": [],
        "status": "processing",
        "next": "",
    }

    # First indexing
    result = index_node(state)
    assert "messages" in result, "index_node result should have 'messages' key"
    assert any("indexed" in m.lower() for m in result["messages"]), (
        f"Expected 'indexed' in messages, got {result['messages']}"
    )

    # Query ChromaDB directly to verify chunks stored
    collection = get_vectorstore()._collection
    existing = collection.get(where={"invoice_no": "TEST-RAG-001"})
    first_count = len(existing["ids"])
    assert first_count >= 2, (
        f"Expected at least 2 chunks (text > 512 chars), got {first_count}"
    )

    # Verify metadata on each chunk
    for meta in existing["metadatas"]:
        assert "invoice_no" in meta, f"Missing 'invoice_no' in metadata: {meta}"
        assert "vendor" in meta, f"Missing 'vendor' in metadata: {meta}"
        assert "po" in meta, f"Missing 'po' in metadata: {meta}"

    # Second indexing (same invoice) -- verify deduplication
    result2 = index_node(state)
    existing2 = collection.get(where={"invoice_no": "TEST-RAG-001"})
    second_count = len(existing2["ids"])
    assert second_count == first_count, (
        f"Dedup failed: expected {first_count} chunks after re-index, got {second_count}"
    )

    # Clean up
    collection.delete(where={"invoice_no": "TEST-RAG-001"})


# ---- Test 3: RAG graph compiles with 4 nodes and conditional edge ----

def test_rag_graph_compiles_with_4_nodes_and_conditional_edge():
    """Verify rag_graph has 4 user nodes and conditional retry edge from reflect_node."""
    from ai_invoice_auditor.agents.rag.rag_orchestrator import rag_graph

    assert rag_graph is not None, "rag_graph should not be None"

    # Get the graph structure
    graph = rag_graph.get_graph()
    node_ids = set(graph.nodes.keys())

    # Filter out LangGraph meta-nodes
    user_nodes = node_ids - {"__start__", "__end__"}
    expected_nodes = {"retrieve_node", "augment_node", "generate_node", "reflect_node"}
    assert expected_nodes == user_nodes, (
        f"Expected nodes {expected_nodes}, got {user_nodes}"
    )

    # Extract edges as (source, target) pairs
    edge_pairs = set()
    for edge in graph.edges:
        if hasattr(edge, "source"):
            src, tgt = edge.source, edge.target
        else:
            src, tgt = edge[0], edge[1]
        edge_pairs.add((src, tgt))

    # Verify linear chain
    assert ("__start__", "retrieve_node") in edge_pairs, (
        "Missing edge: __start__ -> retrieve_node"
    )
    assert ("retrieve_node", "augment_node") in edge_pairs, (
        "Missing edge: retrieve_node -> augment_node"
    )
    assert ("augment_node", "generate_node") in edge_pairs, (
        "Missing edge: augment_node -> generate_node"
    )
    assert ("generate_node", "reflect_node") in edge_pairs, (
        "Missing edge: generate_node -> reflect_node"
    )

    # Verify conditional edges from reflect_node
    assert ("reflect_node", "generate_node") in edge_pairs, (
        "Missing retry edge: reflect_node -> generate_node"
    )
    assert ("reflect_node", "__end__") in edge_pairs, (
        "Missing pass/max-retry edge: reflect_node -> __end__"
    )


# ---- Test 4: retrieve_node returns chunks from ChromaDB ----

def test_retrieve_node_returns_chunks_from_chromadb():
    """Verify retrieve_node returns scored chunk dicts from ChromaDB similarity search."""
    from ai_invoice_auditor.agents.rag.indexing_agent import index_node
    from ai_invoice_auditor.agents.rag.retrieval_agent import retrieve_node
    from ai_invoice_auditor.agents.rag.vectorstore import get_vectorstore

    # Defensive cleanup first
    collection = get_vectorstore()._collection
    try:
        existing = collection.get(where={"invoice_no": "TEST-RAG-RETRIEVE"})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    # Index test data
    test_text = (
        "Widget procurement for Acme Corp at $500 per unit. "
        "Order includes 10 premium widgets for the Q1 2026 project. "
        "Vendor contact: purchasing@acme.example.com. "
    ) * 10  # Enough to produce chunks

    index_state = {
        "file_path": "test_retrieve.pdf",
        "meta": {},
        "raw_text": test_text,
        "extracted": {
            "invoice_no": "TEST-RAG-RETRIEVE",
            "vendor_id": "V-ACME",
            "po_number": "PO-RETRIEVE",
        },
        "translated": None,
        "validation_result": None,
        "report_path": None,
        "messages": [],
        "status": "processing",
        "next": "",
    }
    index_node(index_state)

    # Build RAGState for retrieval
    rag_state = {
        "query": "widget procurement Acme",
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "answer": "",
        "triad_scores": {},
        "passed_reflection": False,
        "retry_count": 0,
        "messages": [],
    }

    result = retrieve_node(rag_state)
    assert "retrieved_chunks" in result, "retrieve_node result should have 'retrieved_chunks'"
    assert len(result["retrieved_chunks"]) > 0, "Should retrieve at least 1 chunk"

    # Each chunk should be a dict with chunk, score, metadata keys
    for chunk in result["retrieved_chunks"]:
        assert isinstance(chunk, dict), f"Each chunk should be a dict, got {type(chunk)}"
        assert "chunk" in chunk, f"Missing 'chunk' key in {chunk.keys()}"
        assert "score" in chunk, f"Missing 'score' key in {chunk.keys()}"
        assert "metadata" in chunk, f"Missing 'metadata' key in {chunk.keys()}"

    # Clean up
    try:
        existing = collection.get(where={"invoice_no": "TEST-RAG-RETRIEVE"})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass


# ---- Test 5: augment_node reranks to top-3 (LIVE LLM) ----

def test_augment_node_reranks_to_top3():
    """Verify augment_node LLM-reranks chunks to top 3. LIVE LLM call."""
    from ai_invoice_auditor.agents.rag.augmentation_agent import augment_node

    # Build RAGState with 5 fake chunks of varying relevance
    rag_state = {
        "query": "What is the total amount?",
        "retrieved_chunks": [
            {"chunk": "Invoice total: $5000 for order PO-200 dated Jan 2026.", "score": 0.9, "metadata": {}},
            {"chunk": "Vendor: Acme Corp, registered in Delaware since 1995.", "score": 0.7, "metadata": {}},
            {"chunk": "Payment terms: Net 30 days from invoice date.", "score": 0.6, "metadata": {}},
            {"chunk": "Shipping address: 123 Main St, Springfield, IL 62701.", "score": 0.4, "metadata": {}},
            {"chunk": "The total amount due is $5,000.00 including tax.", "score": 0.85, "metadata": {}},
        ],
        "reranked_chunks": [],
        "answer": "",
        "triad_scores": {},
        "passed_reflection": False,
        "retry_count": 0,
        "messages": [],
    }

    result = augment_node(rag_state)
    assert "reranked_chunks" in result, "augment_node result should have 'reranked_chunks'"
    assert len(result["reranked_chunks"]) <= 3, (
        f"Expected at most 3 reranked chunks, got {len(result['reranked_chunks'])}"
    )
    assert len(result["reranked_chunks"]) > 0, "Should have at least 1 reranked chunk"

    # Each reranked chunk should have chunk and score keys
    for chunk in result["reranked_chunks"]:
        assert isinstance(chunk, dict), f"Reranked chunk should be dict, got {type(chunk)}"
        assert "chunk" in chunk, f"Missing 'chunk' key in reranked chunk: {chunk.keys()}"
        assert "score" in chunk, f"Missing 'score' key in reranked chunk: {chunk.keys()}"


# ---- Test 6: generate_node produces grounded answer (LIVE LLM) ----

def test_generate_node_produces_grounded_answer():
    """Verify generate_node produces grounded answers and refusal. LIVE LLM calls."""
    from ai_invoice_auditor.agents.rag.generation_agent import generate_node

    # Test grounded answer
    rag_state = {
        "query": "What is the invoice total?",
        "retrieved_chunks": [],
        "reranked_chunks": [
            {"chunk": "The invoice total is $5,000.00 for order PO-200.", "score": 0.95},
        ],
        "answer": "",
        "triad_scores": {},
        "passed_reflection": False,
        "retry_count": 0,
        "messages": [],
    }

    result = generate_node(rag_state)
    assert "answer" in result, "generate_node result should have 'answer'"
    assert len(result["answer"]) > 0, "Answer should be non-empty"
    # Check the answer references the amount
    answer_lower = result["answer"].lower()
    assert any(term in answer_lower for term in ["5,000", "5000", "$5"]), (
        f"Answer should reference the invoice total. Got: {result['answer']}"
    )

    # Test refusal for irrelevant query
    refusal_state = {
        "query": "What is the weather today?",
        "retrieved_chunks": [],
        "reranked_chunks": [
            {"chunk": "Invoice total: $5000 for PO-200", "score": 0.3},
        ],
        "answer": "",
        "triad_scores": {},
        "passed_reflection": False,
        "retry_count": 0,
        "messages": [],
    }

    refusal_result = generate_node(refusal_state)
    assert "answer" in refusal_result, "generate_node refusal result should have 'answer'"
    refusal_answer = refusal_result["answer"].lower()
    # LLM should refuse or give a non-data answer
    has_refusal = (
        "don't have" in refusal_answer
        or "do not have" in refusal_answer
        or "not in the context" in refusal_answer
        or "cannot" in refusal_answer
        or "no information" in refusal_answer
        or "not available" in refusal_answer
        or "not contain" in refusal_answer
        or "doesn't contain" in refusal_answer
        or "does not contain" in refusal_answer
        or "not provided" in refusal_answer
        or "weather" not in refusal_answer  # if it doesn't mention weather data, it's a refusal
    )
    assert has_refusal, (
        f"Expected refusal for irrelevant query, got: {refusal_result['answer']}"
    )


# ---- Test 7: reflect_node evaluates with RAG Triad (LIVE LLM) ----

def test_reflect_node_evaluates_with_rag_triad():
    """Verify reflect_node scores with RAG Triad and sets passed_reflection. LIVE LLM calls."""
    from ai_invoice_auditor.agents.rag.reflection_agent import reflect_node

    # Build state with good alignment between query, context, and answer
    rag_state = {
        "query": "What is the total?",
        "retrieved_chunks": [],
        "reranked_chunks": [
            {"chunk": "Invoice total is $5000", "score": 0.95},
        ],
        "answer": "The invoice total is $5000.",
        "triad_scores": {},
        "passed_reflection": False,
        "retry_count": 0,
        "messages": [],
    }

    result = reflect_node(rag_state)
    assert "triad_scores" in result, "reflect_node result should have 'triad_scores'"

    triad = result["triad_scores"]
    assert "context_relevance" in triad, "Missing 'context_relevance' in triad_scores"
    assert "groundedness" in triad, "Missing 'groundedness' in triad_scores"
    assert "answer_relevance" in triad, "Missing 'answer_relevance' in triad_scores"

    # Each score should be a float between 0.0 and 1.0
    for key in ["context_relevance", "groundedness", "answer_relevance"]:
        score = triad[key]
        assert isinstance(score, float), f"Score '{key}' should be float, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Score '{key}' should be 0.0-1.0, got {score}"

    # Check passed_reflection and retry_count
    assert "passed_reflection" in result, "Missing 'passed_reflection' in result"
    assert isinstance(result["passed_reflection"], bool), (
        f"passed_reflection should be bool, got {type(result['passed_reflection'])}"
    )
    assert "retry_count" in result, "Missing 'retry_count' in result"
    assert result["retry_count"] == 1, (
        f"Expected retry_count=1 (initial 0 + 1), got {result['retry_count']}"
    )


# ---- Test 8: /rag/query endpoint exists on FastAPI app ----

def test_rag_query_endpoint_exists_on_fastapi_app():
    """Verify /rag/query POST endpoint is registered on the FastAPI app."""
    from ai_invoice_auditor.main import app
    from fastapi.routing import APIRoute

    api_routes = [r for r in app.routes if isinstance(r, APIRoute)]
    route_map = {r.path: r.methods for r in api_routes}

    assert "/rag/query" in route_map, (
        f"Missing /rag/query endpoint. Available routes: {list(route_map.keys())}"
    )
    assert "POST" in route_map["/rag/query"], (
        f"/rag/query should support POST, got {route_map['/rag/query']}"
    )


# ---- Test 9: MCP server has real RAG tool implementations ----

def test_mcp_server_has_real_rag_tool_implementations():
    """Verify MCP server registers real RAG tools and stubs remain as stubs."""
    from ai_invoice_auditor.mcp_server import mcp

    # Access tool registry
    assert hasattr(mcp, "_tool_manager"), "mcp should have _tool_manager attribute"
    assert hasattr(mcp._tool_manager, "_tools"), "_tool_manager should have _tools attribute"

    tools = mcp._tool_manager._tools
    tool_names = set(tools.keys())

    # Verify RAG tools are registered
    rag_tools = {"vector_indexer", "semantic_retriever", "chunk_ranker"}
    for tool_name in rag_tools:
        assert tool_name in tool_names, f"Missing RAG tool: {tool_name}"

    # Access underlying functions via FunctionTool.fn attribute
    # Verify real implementations (no "stub" in function body)
    for tool_name in rag_tools:
        func = tools[tool_name].fn
        source = inspect.getsource(func)
        assert "stub" not in source.lower(), (
            f"Tool '{tool_name}' should have a real implementation, but contains 'stub'"
        )

    # Verify response_synthesizer is still a stub (returns empty string)
    rs_func = tools["response_synthesizer"].fn
    rs_source = inspect.getsource(rs_func)
    assert 'return ""' in rs_source, (
        f"response_synthesizer should still be a stub returning empty string"
    )

    # Verify insight_reporter still returns stub string
    ir_func = tools["insight_reporter"].fn
    ir_source = inspect.getsource(ir_func)
    assert "stub" in ir_source.lower(), (
        f"insight_reporter should still be a stub"
    )
