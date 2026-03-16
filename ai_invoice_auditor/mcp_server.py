"""FastMCP server -- registers all 10 project tools as MCP-discoverable endpoints.

8 real tool wrappers call through to existing implementations.
2 stub tool wrappers (insight_reporter, response_synthesizer) remain as placeholders.

Usage:
    from ai_invoice_auditor.mcp_server import mcp
"""

from fastmcp import FastMCP

from ai_invoice_auditor.tools.invoice_watcher import invoice_watcher_tool
from ai_invoice_auditor.tools.data_harvester import data_harvester_tool
from ai_invoice_auditor.tools.lang_bridge import lang_bridge_tool
from ai_invoice_auditor.tools.completeness_checker import data_completeness_checker_tool
from ai_invoice_auditor.tools.business_validator import business_validation_tool

mcp = FastMCP("invoice-auditor")


# ---------------------------------------------------------------------------
# 5 Real Tool Wrappers (call existing implementations)
# ---------------------------------------------------------------------------


@mcp.tool()
def invoice_watcher(directory: str) -> list[dict]:
    """Scan directory for unprocessed invoice+meta file pairs."""
    return invoice_watcher_tool(directory)


@mcp.tool()
def data_harvester(file_path: str) -> str:
    """Extract raw text from PDF, DOCX, or PNG invoice file."""
    return data_harvester_tool(file_path)


@mcp.tool()
def lang_bridge(text: str, source_language: str) -> dict:
    """Translate non-English text to English with confidence score."""
    return lang_bridge_tool(text, source_language)


@mcp.tool()
def completeness_checker(extracted: dict) -> dict:
    """Check extracted invoice data for missing required fields."""
    return data_completeness_checker_tool(extracted)


@mcp.tool()
def business_validator(extracted: dict) -> dict:
    """Validate invoice against ERP data with concurrent lookups."""
    return business_validation_tool(extracted)


# ---------------------------------------------------------------------------
# 2 Stub Tool Wrappers + 3 Real RAG Tool Wrappers
# ---------------------------------------------------------------------------


@mcp.tool()
def insight_reporter(state: dict) -> str:
    """Generate HTML audit report from validation results."""
    return "stub -- report generation via report_node"


@mcp.tool()
def vector_indexer(text: str, metadata: dict) -> bool:
    """Chunk and embed invoice text into ChromaDB."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from ai_invoice_auditor.agents.rag.vectorstore import get_vectorstore

    if not text.strip():
        return False

    vectorstore = get_vectorstore()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(text)
    vectorstore.add_texts(texts=chunks, metadatas=[metadata] * len(chunks))
    return True


@mcp.tool()
def semantic_retriever(query: str, k: int = 10) -> list[dict]:
    """Semantic similarity search against indexed invoices."""
    from ai_invoice_auditor.agents.rag.vectorstore import get_vectorstore

    vectorstore = get_vectorstore()
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
    except Exception:
        return []

    return [
        {
            "chunk": doc.page_content,
            "score": float(score),
            "metadata": doc.metadata,
        }
        for doc, score in results
    ]


@mcp.tool()
def chunk_ranker(query: str, chunks: list[str]) -> list[dict]:
    """LLM-based reranking of retrieved chunks."""
    import json
    from ai_invoice_auditor.llm import get_llm

    if not chunks:
        return []

    _, chat_model, _ = get_llm()

    numbered = "\n".join(f"{i + 1}. {c[:500]}" for i, c in enumerate(chunks))
    prompt = (
        f'You are a relevance ranking assistant.\n'
        f'Query: "{query}"\n\n'
        f'Rank the following chunks by relevance. Return a JSON array of objects '
        f'for the top 3 most relevant, each with "index" (1-based) and "score" (0.0-1.0).\n'
        f'Return ONLY the JSON array.\n\nChunks:\n{numbered}'
    )

    response = chat_model.invoke(prompt)
    content = response.content.strip()

    # Strip markdown code fences
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if "```" in content:
            content = content.rsplit("```", 1)[0].strip()

    try:
        rankings = json.loads(content)
        result = []
        for entry in rankings[:3]:
            idx = entry["index"] - 1
            if 0 <= idx < len(chunks):
                result.append({"chunk": chunks[idx], "score": float(entry["score"])})
        return result
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return [{"chunk": c, "score": 0.5} for c in chunks[:3]]


@mcp.tool()
def response_synthesizer(query: str, context_chunks: list[str]) -> str:
    """Generate grounded answer from context chunks."""
    return ""
