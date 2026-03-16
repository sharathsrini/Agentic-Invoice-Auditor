import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ai_invoice_auditor.guardrails import mask_pii
from ai_invoice_auditor.models.state import InvoiceState

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "reports"


def report_node(state: InvoiceState) -> dict:
    """
    LangGraph node: Generates an HTML audit report using Jinja2.

    Renders the report.html.j2 template with invoice data, validation
    results, and audit trail messages. Writes the HTML file to
    outputs/reports/{invoice_no}_report.html.
    """
    try:
        logger.info(f"[report_node] Processing: {state.get('file_path', 'unknown')}")

        env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
        template = env.get_template("report.html.j2")

        extracted = state.get("extracted", {})
        invoice_no = extracted.get("invoice_no", "UNKNOWN")

        html = template.render(
            invoice_no=invoice_no,
            extracted=extracted,
            translated=state.get("translated", {}),
            validation_result=state.get("validation_result", {}),
            status=state.get("status", "unknown"),
            messages=state.get("messages", []),
        )

        # Mask PII (phone numbers, IBANs) in HTML output before writing
        html = mask_pii(html)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / f"{invoice_no}_report.html"
        report_path.write_text(html, encoding="utf-8")

        logger.info(f"[report_node] Generated report: {report_path}")

        return {
            "report_path": str(report_path),
            "messages": [f"report_node: generated {report_path}"],
        }
    except FileNotFoundError as e:
        logger.error(f"[report_node] File not found: {e}")
        return {
            "status": "reject",
            "messages": [f"ERROR [report_node]: file not found -- {e}"],
        }
    except Exception as e:
        logger.error(f"[report_node] Failed: {e}")
        return {
            "status": "flag",
            "messages": [f"ERROR [report_node]: {e}"],
        }
