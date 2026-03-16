"""Extractor agent -- LLM-based structured extraction with hallucination
verification, fuzzy vendor resolution, and PO number regex fallback.

The extract_node is the core extraction LangGraph node. It takes raw invoice
files, extracts text via data_harvester, structures fields via LLM
with_structured_output, then verifies against source text to catch
hallucinations. Vendor IDs are resolved via rapidfuzz fuzzy matching against
ERP vendor data, and PO numbers fall back to regex on email subject metadata.
"""

import json
import logging
import re
from pathlib import Path

from ai_invoice_auditor.guardrails import sanitize_text
from ai_invoice_auditor.models.state import InvoiceState

logger = logging.getLogger(__name__)

# Module-level vendor cache (loaded once on first call)
_vendors: list[dict] | None = None


def _load_vendors() -> list[dict]:
    """Load vendors.json from ERP mock data. Cached at module level."""
    global _vendors
    if _vendors is not None:
        return _vendors
    vendors_path = Path(__file__).resolve().parent.parent.parent / "data" / "ERP_mockdata" / "vendors.json"
    with open(vendors_path, "r", encoding="utf-8") as f:
        _vendors = json.load(f)
    logger.info("[extractor] Loaded %d vendors from %s", len(_vendors), vendors_path)
    return _vendors


def _verify_extraction(extracted: dict, raw_text: str) -> dict:
    """Verify extracted fields against source text to catch LLM hallucinations.

    Fields that cannot be found in the raw source text are cleared to None.
    This prevents the LLM from inventing invoice numbers, PO numbers, or
    currency codes that do not exist in the original document.

    Args:
        extracted: Dict of extracted invoice fields from LLM.
        raw_text: The original raw text from the invoice file.

    Returns:
        The extracted dict with hallucinated fields cleared to None.
    """
    # Check invoice_no: must appear in raw text (case-sensitive)
    if extracted.get("invoice_no") is not None:
        if extracted["invoice_no"] not in raw_text:
            logger.warning(
                "[verify_extraction] Clearing hallucinated invoice_no='%s' -- not found in source text",
                extracted["invoice_no"],
            )
            extracted["invoice_no"] = None

    # Check po_number: must appear in raw text (case-sensitive)
    if extracted.get("po_number") is not None:
        if extracted["po_number"] not in raw_text:
            logger.warning(
                "[verify_extraction] Clearing hallucinated po_number='%s' -- not found in source text",
                extracted["po_number"],
            )
            extracted["po_number"] = None

    # Check currency: at least one currency indicator must exist in raw text
    if extracted.get("currency") is not None:
        currency_indicators = [
            "$", "\u20ac", "\u00a3", "\u20b9",  # symbols: $, euro, pound, rupee
            "USD", "EUR", "GBP", "INR",
            extracted["currency"],
        ]
        has_currency = any(indicator in raw_text for indicator in currency_indicators)
        if not has_currency:
            logger.warning(
                "[verify_extraction] Clearing hallucinated currency='%s' -- no currency indicator in source text",
                extracted["currency"],
            )
            extracted["currency"] = None

    # Check total_amount: log warning but do NOT clear (LLM may reformat numbers)
    if extracted.get("total_amount") is not None:
        amount_str = str(extracted["total_amount"])
        if amount_str not in raw_text:
            logger.warning(
                "[verify_extraction] total_amount='%s' not found verbatim in source text (may be reformatted)",
                amount_str,
            )

    return extracted


def _resolve_vendor_id(vendor_name: str | None, sender_email: str | None) -> str | None:
    """Resolve vendor name to vendor_id using fuzzy matching against ERP data.

    Args:
        vendor_name: Vendor name extracted from the invoice (may be None).
        sender_email: Sender email from invoice metadata (may be None).

    Returns:
        Matched vendor_id (e.g. 'VEND-001') or None if no match found.
    """
    from rapidfuzz import process, fuzz

    vendors = _load_vendors()
    choices_dict = {v["vendor_id"]: v["vendor_name"] for v in vendors}

    # Attempt 1: Fuzzy match on vendor name
    if vendor_name is not None:
        match = process.extractOne(
            vendor_name,
            choices_dict,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=60,
        )
        if match is not None:
            # match is (matched_value, score, key)
            vendor_id = match[2]
            logger.info(
                "[vendor_resolve] Fuzzy matched '%s' -> %s (score: %.1f)",
                vendor_name, vendor_id, match[1],
            )
            return vendor_id

    # Attempt 2: Email domain fallback
    if sender_email is not None:
        try:
            domain = sender_email.split("@")[1].split(".")[0].lower()
            for v in vendors:
                # Compare domain against vendor name with spaces removed
                vendor_name_compact = v["vendor_name"].lower().replace(" ", "")
                if domain in vendor_name_compact or domain in v["vendor_name"].lower():
                    logger.info(
                        "[vendor_resolve] Email domain '%s' matched vendor '%s' (%s)",
                        domain, v["vendor_name"], v["vendor_id"],
                    )
                    return v["vendor_id"]
        except (IndexError, AttributeError):
            pass

    logger.warning("[vendor_resolve] No match for vendor_name='%s', email='%s'", vendor_name, sender_email)
    return None


def _resolve_po_number(extracted_po: str | None, meta: dict) -> str | None:
    """Resolve PO number, falling back to regex on meta subject if LLM missed it.

    Args:
        extracted_po: PO number from LLM extraction (may be None).
        meta: Invoice metadata dict (may contain 'subject' key).

    Returns:
        PO number string (e.g. 'PO-1001') or None.
    """
    if extracted_po is not None:
        return extracted_po

    subject = meta.get("subject", "")
    match = re.search(r"PO-\d{4}", subject)
    if match:
        po = match.group(0)
        logger.info("[po_resolve] Regex extracted '%s' from meta subject", po)
        return po

    return None


def extract_node(state: InvoiceState) -> dict:
    """LangGraph node: Extracts structured data from raw invoice text.

    Performs 5 steps:
    1. Extract raw text via data_harvester_tool
    2. LLM structured extraction via with_structured_output(ExtractedInvoiceHeader)
    3. Verify extraction against source text (hallucination check)
    4. Resolve vendor ID via fuzzy match
    5. Resolve PO number via regex fallback on meta subject

    Returns partial dict with raw_text, extracted, and messages.
    """
    try:
        file_path = state.get("file_path", "")
        meta = state.get("meta", {})
        logger.info("[extract_node] Processing: %s", file_path)

        # Step 1: Extract raw text
        from ai_invoice_auditor.tools.data_harvester import data_harvester_tool

        raw_text = data_harvester_tool(file_path)

        if not raw_text or not raw_text.strip():
            logger.warning("[extract_node] Empty text extracted from %s", file_path)
            return {
                "raw_text": raw_text,
                "status": "flag",
                "messages": [f"WARNING [extract_node]: empty text from {file_path}"],
            }

        # Step 1.5: Sanitize text (prompt injection guard)
        raw_text = sanitize_text(raw_text)

        # Step 2: LLM structured extraction
        from ai_invoice_auditor.llm import get_llm
        from ai_invoice_auditor.models.invoice import ExtractedInvoiceHeader

        _, chat_model, _ = get_llm()
        extractor = chat_model.with_structured_output(ExtractedInvoiceHeader)
        result = extractor.invoke(
            f"Extract all invoice fields from this text. Return None for any field not clearly present in the text.\n\n{raw_text}"
        )
        extracted = result.model_dump()

        # Step 3: Verify extraction (hallucination check)
        extracted = _verify_extraction(extracted, raw_text)

        # Step 4: Vendor ID resolution
        vendor_id = extracted.get("vendor_id")
        if vendor_id is None or not re.match(r"^VEND-\d{3}$", str(vendor_id)):
            # Use the LLM-extracted vendor_id as a vendor name hint for fuzzy matching
            vendor_name_hint = vendor_id if vendor_id and not re.match(r"^VEND-\d{3}$", str(vendor_id)) else None
            resolved_vid = _resolve_vendor_id(
                vendor_name_hint,
                meta.get("sender"),
            )
            if resolved_vid is not None:
                extracted["vendor_id"] = resolved_vid

        # Step 5: PO number resolution (regex fallback)
        extracted["po_number"] = _resolve_po_number(extracted.get("po_number"), meta)

        logger.info("[extract_node] Extraction complete for %s", file_path)

        return {
            "raw_text": raw_text,
            "extracted": extracted,
            "messages": [f"extract_node: completed extraction for {file_path}"],
        }

    except FileNotFoundError as e:
        logger.error("[extract_node] File not found: %s", e)
        return {
            "status": "reject",
            "messages": [f"ERROR [extract_node]: file not found -- {e}"],
        }
    except Exception as e:
        logger.error("[extract_node] Failed: %s", e)
        result = {
            "status": "flag",
            "messages": [f"ERROR [extract_node]: {e}"],
        }
        # Preserve raw_text if it was captured before the error
        if "raw_text" in dir() and raw_text is not None:
            result["raw_text"] = raw_text
        return result
