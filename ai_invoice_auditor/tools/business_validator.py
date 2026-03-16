"""Business validation tool with concurrent ERP fetches.

Compares invoice line items against purchase order data from the ERP mock API,
using ThreadPoolExecutor for concurrent vendor + PO lookups (FOUND-04).
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import httpx

from ai_invoice_auditor.config import get_config

logger = logging.getLogger(__name__)


def compare_line_items(
    invoice_items: list[dict],
    po_items: list[dict],
    tolerances: dict,
) -> list[dict]:
    """Compare invoice line items against PO line items for discrepancies.

    Matches by item_code (not list position). Returns a list of discrepancy
    dicts for any price or quantity deviations exceeding configured tolerances.

    Args:
        invoice_items: Line items from the extracted invoice.
        po_items: Line items from the ERP purchase order.
        tolerances: Dict with price_difference_percent and
            quantity_difference_percent keys.

    Returns:
        List of discrepancy dicts, each with item_code, field, invoice_value,
        erp_value, deviation_pct (percentage), and breaches_tolerance.
    """
    price_tol = tolerances["price_difference_percent"] / 100
    qty_tol = tolerances["quantity_difference_percent"] / 100

    # Build PO lookup by item_code (NOT positional zip)
    po_lookup = {item["item_code"]: item for item in po_items}

    discrepancies: list[dict] = []

    for inv_item in invoice_items:
        item_code = inv_item.get("item_code")
        if item_code is None or item_code not in po_lookup:
            continue

        po_item = po_lookup[item_code]

        # --- Price comparison ---
        inv_price = inv_item.get("unit_price")
        erp_price = po_item.get("unit_price")

        if inv_price is not None and erp_price is not None and erp_price > 0:
            deviation_pct = abs(inv_price - erp_price) / erp_price
            if deviation_pct > price_tol:
                discrepancies.append({
                    "item_code": item_code,
                    "field": "unit_price",
                    "invoice_value": inv_price,
                    "erp_value": erp_price,
                    "deviation_pct": round(deviation_pct * 100, 1),
                    "breaches_tolerance": True,
                })

        # --- Quantity comparison ---
        inv_qty = inv_item.get("qty")
        erp_qty = po_item.get("qty")

        if inv_qty is not None and erp_qty is not None and erp_qty > 0:
            qty_deviation_pct = abs(inv_qty - erp_qty) / erp_qty
            if qty_deviation_pct > qty_tol:
                discrepancies.append({
                    "item_code": item_code,
                    "field": "qty",
                    "invoice_value": inv_qty,
                    "erp_value": erp_qty,
                    "deviation_pct": round(qty_deviation_pct * 100, 1),
                    "breaches_tolerance": True,
                })

    return discrepancies


def _fetch_erp_data(
    po_number: str, vendor_id: str, erp_base_url: str
) -> tuple[httpx.Response, httpx.Response]:
    """Fetch PO and vendor data from ERP concurrently using ThreadPoolExecutor.

    Each thread creates its own httpx.Client for thread safety (httpx.Client
    is NOT thread-safe). Uses ThreadPoolExecutor instead of asyncio.gather
    because the caller (business_validation_tool) is synchronous.

    Args:
        po_number: Purchase order number to look up.
        vendor_id: Vendor ID to look up.
        erp_base_url: Base URL of the ERP API (e.g. http://localhost:8000).

    Returns:
        Tuple of (po_response, vendor_response).
    """

    def fetch_po() -> httpx.Response:
        with httpx.Client() as client:
            return client.get(f"{erp_base_url}/erp/po/{po_number}")

    def fetch_vendor() -> httpx.Response:
        with httpx.Client() as client:
            return client.get(f"{erp_base_url}/erp/vendor/{vendor_id}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        po_future = executor.submit(fetch_po)
        vendor_future = executor.submit(fetch_vendor)
        po_response = po_future.result()
        vendor_response = vendor_future.result()

    return po_response, vendor_response


def business_validation_tool(
    extracted: dict,
    config: dict | None = None,
    erp_base_url: str = "http://localhost:8000",
) -> dict:
    """Validate invoice against ERP data with concurrent vendor+PO fetches.

    Fetches vendor and PO data concurrently, compares line items against
    PO, and determines validation status (auto_approve / flag / manual_review).

    Args:
        extracted: Dict of extracted invoice fields including po_number,
            vendor_id, line_items, and optionally translation_confidence.
        config: Optional rules config dict. Loaded from rules.yaml if None.
        erp_base_url: Base URL for ERP API calls.

    Returns:
        Dict with status, discrepancies list, vendor data, and PO data.
    """
    if config is None:
        config = get_config()

    po_number = extracted.get("po_number")
    vendor_id = extracted.get("vendor_id")

    # Fetch vendor and PO data concurrently (FOUND-04)
    logger.info(
        "Fetching ERP data concurrently: PO=%s, Vendor=%s",
        po_number,
        vendor_id,
    )
    po_response, vendor_response = _fetch_erp_data(
        po_number, vendor_id, erp_base_url
    )

    # Check for non-200 responses
    if po_response.status_code != 200:
        logger.warning("PO lookup failed: %s", po_response.status_code)
        return {
            "status": "flag",
            "reason": f"PO lookup failed with status {po_response.status_code}",
            "discrepancies": [],
            "vendor": None,
            "po": None,
        }

    if vendor_response.status_code != 200:
        logger.warning("Vendor lookup failed: %s", vendor_response.status_code)
        return {
            "status": "flag",
            "reason": f"Vendor lookup failed with status {vendor_response.status_code}",
            "discrepancies": [],
            "vendor": None,
            "po": None,
        }

    po_data = po_response.json()
    vendor_data = vendor_response.json()

    # Compare line items
    tolerances = config["tolerances"]
    invoice_items = extracted.get("line_items", [])
    po_items = po_data.get("line_items", [])
    discrepancies = compare_line_items(invoice_items, po_items, tolerances)

    # Determine status
    has_breaching = any(d["breaches_tolerance"] for d in discrepancies)

    if has_breaching:
        status = "manual_review"
    elif not discrepancies:
        # Check auto-approve threshold
        translation_confidence = extracted.get("translation_confidence", 1.0)
        auto_approve_threshold = config.get(
            "validation_policies", {}
        ).get("auto_approve_confidence_threshold", 0.95)

        if translation_confidence >= auto_approve_threshold:
            status = "auto_approve"
        else:
            status = "flag"
    else:
        status = "flag"

    logger.info(
        "Business validation: %d discrepancies, status=%s",
        len(discrepancies),
        status,
    )

    return {
        "status": status,
        "discrepancies": discrepancies,
        "vendor": vendor_data,
        "po": po_data,
    }
