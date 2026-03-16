"""FastAPI application entry point with MCP server startup and LangFuse observability.

Registers:
    - /erp/* -- ERP mock data endpoints (vendors, POs, SKUs)
    - /invoice/* -- Pipeline invocation, status, resume, and report endpoints
    - /rag/* -- RAG query endpoint for grounded invoice Q&A
    - /health -- Health check

Lifespan:
    - Loads MCP server for in-process tool discovery (ORCH-08)
    - Initializes LangFuse observability (graceful degradation when unconfigured)

Run: uvicorn ai_invoice_auditor.main:app --reload
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Load .env file before any other imports that need env vars
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip().strip('"').strip("'"))

from fastapi import FastAPI

from ai_invoice_auditor.api.erp_router import router as erp_router
from ai_invoice_auditor.api.invoice_router import router as invoice_router
from ai_invoice_auditor.api.rag_router import router as rag_router
from ai_invoice_auditor.observability import get_langfuse_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --- MCP Server Startup (ORCH-08) ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler -- startup and shutdown hooks."""
    # Startup: import and load MCP server
    try:
        from ai_invoice_auditor.mcp_server import mcp as mcp_server

        logger.info("MCP server '%s' loaded with tools registered", mcp_server.name)
    except ImportError as e:
        logger.warning("MCP server not available: %s", e)

    # Log active LLM provider
    try:
        from ai_invoice_auditor.llm import get_llm

        provider_name, _, _ = get_llm()
        logger.info("LLM provider active: %s", provider_name)
    except Exception as e:
        logger.warning("LLM provider not available at startup: %s", e)

    # Log LangFuse status
    langfuse_handler = get_langfuse_handler()
    langfuse_status = "configured" if langfuse_handler else "disabled"
    logger.info(
        "AI Invoice Auditor started (LangFuse: %s)", langfuse_status
    )

    yield  # App runs here

    # Shutdown
    logger.info("Shutting down AI Invoice Auditor")


app = FastAPI(
    title="AI Invoice Auditor",
    description="Agentic AI-powered multilingual invoice validation system",
    version="0.1.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(erp_router)
app.include_router(invoice_router)
app.include_router(rag_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
