import logging

from langgraph.types import Command, interrupt

from ai_invoice_auditor.guardrails import block_low_confidence_auto_approval
from ai_invoice_auditor.models.state import InvoiceState
from ai_invoice_auditor.mcp_client import call_business_validator

logger = logging.getLogger(__name__)


def business_validate_node(state: InvoiceState) -> dict:
    """
    LangGraph node: Validates invoice against ERP data (vendor, PO, prices).

    Calls call_business_validator (MCP client wrapper) on extracted data. When the result status
    is "manual_review", triggers interrupt() to pause the graph and expose
    discrepancies to a human reviewer. On resume:
      - decision="approve" -> sets status to auto_approve
      - decision="reject"  -> sets status to reject
      - decision="edit"    -> merges corrections into extracted and returns
        Command(goto="data_validate_node") for graph-level re-routing

    NOTE: The node re-executes from the beginning on resume (LangGraph
    interrupt semantics). The call_business_validator call is idempotent
    (same extracted data -> same result), so this is safe.
    """
    try:
        logger.info(
            f"[business_validate_node] Processing: {state.get('file_path', 'unknown')}"
        )
        extracted = state.get("extracted", {})
        result = call_business_validator(extracted)

        # XCUT-02(b): Confidence guardrail -- block auto-approve when
        # translation quality is uncertain (defense-in-depth)
        translation_confidence = state.get("translated", {}).get("translation_confidence", 1.0)
        result["status"] = block_low_confidence_auto_approval(
            result["status"], translation_confidence
        )

        if result["status"] == "manual_review":
            # Pause graph -- expose discrepancies to human reviewer
            human_decision = interrupt({
                "invoice_no": extracted.get("invoice_no"),
                "discrepancies": result.get("discrepancies", []),
                "message": "Discrepancies found. Approve, reject, or edit.",
            })

            decision = human_decision.get("decision", "reject")

            if decision == "approve":
                result["status"] = "auto_approve"
                result["reviewer_decision"] = "approve"
            elif decision == "reject":
                result["status"] = "reject"
                result["reviewer_decision"] = "reject"
            elif decision == "edit":
                # Merge corrections into extracted and re-route to data_validate_node
                corrections = human_decision.get("corrections", {})
                updated_extracted = {**extracted, **corrections}
                return Command(
                    goto="data_validate_node",
                    update={
                        "extracted": updated_extracted,
                        "status": "flag",
                        "messages": [
                            "biz_validate_node: human edit applied, "
                            "re-routing to data_validate_node"
                        ],
                    },
                )

        return {
            "validation_result": result,
            "status": result["status"],
            "messages": [f"biz_validate_node: status={result['status']}"],
        }
    except FileNotFoundError as e:
        logger.error(f"[business_validate_node] File not found: {e}")
        return {
            "status": "reject",
            "messages": [f"ERROR [business_validate_node]: file not found -- {e}"],
        }
    except Exception as e:
        # Let GraphInterrupt propagate — it's how interrupt() signals HITL pause
        from langgraph.errors import GraphInterrupt
        if isinstance(e, GraphInterrupt):
            raise
        logger.error(f"[business_validate_node] Failed: {e}")
        return {
            "status": "flag",
            "messages": [f"ERROR [business_validate_node]: {e}"],
        }
