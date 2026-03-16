import logging
import os
from pathlib import Path

from ai_invoice_auditor.models.state import InvoiceState
from ai_invoice_auditor.tools.invoice_watcher import invoice_watcher_tool

logger = logging.getLogger(__name__)


def monitor_node(state: InvoiceState) -> dict:
    """
    LangGraph node: Validates invoice file existence from state.

    In the standard single-invoice pipeline, the orchestrator populates
    file_path and meta before graph invocation. The monitor_node validates
    the file exists and is accessible.

    Fallback: if file_path is empty/missing, scans data/incoming/ via
    invoice_watcher_tool and populates state with the first unprocessed invoice.
    """
    try:
        file_path = state.get("file_path", "")
        meta = state.get("meta", {})

        # If file_path is provided, validate it exists
        if file_path:
            fp = Path(file_path)
            if not fp.exists():
                raise FileNotFoundError(
                    f"Invoice file not found: {file_path}"
                )
            logger.info(
                "[monitor_node] Validated file exists: %s", file_path
            )
            return {
                "file_path": file_path,
                "meta": meta,
                "messages": [
                    f"monitor_node: validated {file_path}"
                ],
            }

        # Fallback: no file_path in state -- scan incoming directory
        incoming_dir = os.getenv("INCOMING_DIR", "data/incoming")
        logger.info(
            "[monitor_node] No file_path in state, scanning %s", incoming_dir
        )
        invoices = invoice_watcher_tool(incoming_dir)

        if not invoices:
            logger.info("[monitor_node] No unprocessed invoices found")
            return {
                "status": "flag",
                "messages": [
                    "monitor_node: no unprocessed invoices found in "
                    f"{incoming_dir}"
                ],
            }

        # Populate state with the first unprocessed invoice
        first = invoices[0]
        logger.info(
            "[monitor_node] Found %d invoice(s), using: %s",
            len(invoices),
            first["file_path"],
        )
        return {
            "file_path": first["file_path"],
            "meta": first["meta"],
            "messages": [
                f"monitor_node: scanned {incoming_dir}, "
                f"found {len(invoices)} invoice(s), "
                f"using {first['file_path']}"
            ],
        }

    except FileNotFoundError as e:
        logger.error("[monitor_node] File not found: %s", e)
        return {
            "status": "reject",
            "messages": [f"ERROR [monitor_node]: file not found -- {e}"],
        }
    except Exception as e:
        logger.error("[monitor_node] Failed: %s", e)
        return {
            "status": "flag",
            "messages": [f"ERROR [monitor_node]: {e}"],
        }
