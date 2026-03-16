import logging

from ai_invoice_auditor.models.state import InvoiceState
from ai_invoice_auditor.mcp_client import call_completeness_checker

logger = logging.getLogger(__name__)


def data_validate_node(state: InvoiceState) -> dict:
    """
    LangGraph node: Validates extracted data completeness against rules.yaml.

    Calls call_completeness_checker (MCP client wrapper) on the extracted dict
    and propagates the validation result and status downstream.
    """
    try:
        logger.info(f"[data_validate_node] Processing: {state.get('file_path', 'unknown')}")
        extracted = state.get("extracted", {})
        result = call_completeness_checker(extracted)

        # Propagate status from tool result; fall back to current state status
        status = result.get("status", state.get("status", ""))

        return {
            "validation_result": result,
            "status": status,
            "messages": [f"data_validate_node: status={result.get('status')}"],
        }
    except FileNotFoundError as e:
        logger.error(f"[data_validate_node] File not found: {e}")
        return {
            "status": "reject",
            "messages": [f"ERROR [data_validate_node]: file not found -- {e}"],
        }
    except Exception as e:
        logger.error(f"[data_validate_node] Failed: {e}")
        return {
            "status": "flag",
            "messages": [f"ERROR [data_validate_node]: {e}"],
        }
