"""FastAPI invoice router -- pipeline invocation, status, resume, and report endpoints.

Endpoints:
    POST /invoice/process          -- Invoke the full pipeline for an invoice
    POST /invoice/{invoice_no}/resume -- Resume an interrupted pipeline with human decision
    GET  /invoice/{invoice_no}     -- Get current pipeline state for an invoice
    GET  /invoice/{invoice_no}/report -- Serve the HTML audit report
"""

import asyncio
import logging
import uuid
from functools import partial

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path

from langgraph.types import Command

from ai_invoice_auditor.observability import get_langfuse_handler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/invoice", tags=["Invoice Pipeline"])

# In-memory tracking of processed invoices for the Streamlit Live Audit tab.
_processed_invoices: dict[str, dict] = {}


def _get_pipeline_graph():
    """Lazy import to avoid circular imports at module load time."""
    from ai_invoice_auditor.agents.orchestrator import pipeline_graph
    return pipeline_graph


def _build_config(thread_id: str) -> dict:
    """Build LangGraph config with thread_id and optional LangFuse callbacks.

    Args:
        thread_id: The thread identifier for LangGraph checkpointing.

    Returns:
        Config dict with configurable.thread_id and callbacks (if LangFuse configured).
    """
    config = {"configurable": {"thread_id": thread_id}}
    handler = get_langfuse_handler()
    if handler is not None:
        config["callbacks"] = [handler]
    return config


@router.post("/process")
async def process_invoice(request: dict):
    """Invoke the full invoice processing pipeline.

    Request body:
        file_path: str -- Path to the invoice file
        meta: dict -- Invoice metadata (invoice_no, language, sender, etc.)

    Returns:
        Pipeline result with status, invoice_no, report_path, and messages.
        Returns 202 if the pipeline is interrupted for human review.
    """
    try:
        file_path = request.get("file_path", "")
        meta = request.get("meta", {})
        invoice_no = meta.get("invoice_no") or f"INV-{uuid.uuid4().hex[:8].upper()}"

        # Build initial state
        state = {
            "file_path": file_path,
            "meta": meta,
            "raw_text": None,
            "extracted": None,
            "translated": None,
            "validation_result": None,
            "report_path": None,
            "messages": [],
            "status": "",
            "next": "",
        }

        # Unique thread_id per invocation (prevents stale checkpoint collisions)
        thread_id = f"invoice-{invoice_no}-{uuid.uuid4().hex[:8]}"
        config = _build_config(thread_id)

        logger.info(
            "[invoice_router] Processing invoice %s (thread=%s)", invoice_no, thread_id
        )

        pipeline_graph = _get_pipeline_graph()

        try:
            from langgraph.errors import GraphInterrupt
        except ImportError:
            GraphInterrupt = None

        # Run pipeline in thread pool so async event loop stays free
        # to serve ERP endpoints (prevents self-deadlock)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, partial(pipeline_graph.invoke, state, config=config)
        )

        # Check if the graph was interrupted (HITL pause)
        # With MemorySaver, interrupt() returns normally with __interrupt__ key
        if "__interrupt__" in result:
            logger.info(
                "[invoice_router] Pipeline interrupted for invoice %s", invoice_no
            )
            # Extract discrepancy data from the interrupt value
            # The interrupt fires INSIDE biz_validate_node, so the business
            # validation result hasn't been written to state yet.
            # The discrepancies are in the interrupt value itself.
            interrupt_data = result["__interrupt__"]
            iv = interrupt_data[0].value if interrupt_data else {}
            validation_result = {
                "status": "manual_review",
                "discrepancies": iv.get("discrepancies", []),
                # Also keep completeness check data
                "completeness": result.get("validation_result"),
            }
            _processed_invoices[invoice_no] = {
                "status": "interrupted",
                "report_path": None,
                "messages": result.get("messages", []) + ["Human review required"],
                "thread_id": thread_id,
                "extracted": result.get("extracted"),
                "validation_result": validation_result,
            }
            return JSONResponse(
                status_code=202,
                content={
                    "status": "interrupted",
                    "invoice_no": invoice_no,
                    "message": "Human review required",
                    "thread_id": thread_id,
                    "messages": result.get("messages", []),
                },
            )

        _processed_invoices[invoice_no] = {
            "status": result.get("status", "unknown"),
            "report_path": result.get("report_path"),
            "messages": result.get("messages", []),
            "thread_id": thread_id,
            "extracted": result.get("extracted"),
            "validation_result": result.get("validation_result"),
        }

        return {
            "status": result.get("status", "unknown"),
            "invoice_no": invoice_no,
            "report_path": result.get("report_path"),
            "messages": result.get("messages", []),
        }

    except Exception as e:
        logger.error("[invoice_router] Process failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{invoice_no}/resume")
async def resume_invoice(invoice_no: str, decision: dict):
    """Resume an interrupted pipeline with a human decision.

    Path params:
        invoice_no: str -- The invoice number to resume

    Request body (decision):
        decision: str -- "approve", "reject", or "edit"
        corrections: dict -- (optional) field corrections for "edit" decision

    Returns:
        Pipeline result with status, report_path, and messages.
    """
    try:
        # Look up thread_id from when the invoice was processed
        stored = _processed_invoices.get(invoice_no, {})
        thread_id = stored.get("thread_id", f"invoice-{invoice_no}")
        config = _build_config(thread_id)

        logger.info(
            "[invoice_router] Resuming invoice %s (thread=%s) with decision: %s",
            invoice_no,
            thread_id,
            decision.get("decision"),
        )

        pipeline_graph = _get_pipeline_graph()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, partial(pipeline_graph.invoke, Command(resume=decision), config=config)
        )

        _processed_invoices[invoice_no] = {
            "status": result.get("status", "unknown"),
            "report_path": result.get("report_path"),
            "messages": result.get("messages", []),
        }

        return {
            "status": result.get("status", "unknown"),
            "invoice_no": invoice_no,
            "report_path": result.get("report_path"),
            "messages": result.get("messages", []),
        }

    except Exception as e:
        logger.error("[invoice_router] Resume failed for %s: %s", invoice_no, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def get_invoice_list():
    """List all processed invoices with their current status."""
    return {
        "invoices": [
            {"invoice_no": k, **v} for k, v in _processed_invoices.items()
        ]
    }


@router.get("/{invoice_no}/report")
async def get_report(invoice_no: str):
    """Serve the HTML audit report for an invoice.

    Returns the generated HTML file from outputs/reports/{invoice_no}_report.html.
    """
    report_path = Path("outputs/reports") / f"{invoice_no}_report.html"

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(str(report_path), media_type="text/html")


@router.get("/{invoice_no}")
async def get_invoice_status(invoice_no: str):
    """Get the current pipeline state for an invoice.

    Uses the LangGraph checkpointer to retrieve the latest state snapshot
    for the given invoice's thread.
    """
    # First check in-memory store (has the correct thread_id)
    stored = _processed_invoices.get(invoice_no)
    if stored is None:
        raise HTTPException(status_code=404, detail="Invoice not found")

    # Try to get full state from LangGraph checkpointer
    thread_id = stored.get("thread_id", f"invoice-{invoice_no}")
    try:
        pipeline_graph = _get_pipeline_graph()
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = pipeline_graph.get_state(config)

        if state_snapshot and state_snapshot.values:
            values = state_snapshot.values
            is_interrupted = stored.get("status") == "interrupted"
            return {
                "invoice_no": invoice_no,
                "status": stored.get("status", values.get("status", "unknown")),
                "messages": values.get("messages", stored.get("messages", [])),
                "report_path": values.get("report_path", stored.get("report_path")),
                # For interrupted invoices, prefer stored validation_result (has discrepancies
                # from interrupt value) over checkpoint's (which is the completeness check)
                "validation_result": stored.get("validation_result") if is_interrupted else values.get("validation_result", stored.get("validation_result")),
                "extracted": values.get("extracted", stored.get("extracted")),
            }
    except Exception as e:
        logger.warning("[invoice_router] Checkpointer lookup failed for %s: %s", invoice_no, e)

    # Fallback to stored data
    return {
        "invoice_no": invoice_no,
        "status": stored.get("status", "unknown"),
        "messages": stored.get("messages", []),
        "report_path": stored.get("report_path"),
        "validation_result": stored.get("validation_result"),
        "extracted": stored.get("extracted"),
    }
