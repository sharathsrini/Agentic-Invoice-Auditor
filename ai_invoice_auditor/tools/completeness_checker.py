"""Data completeness checker tool.

Validates that all required header and line-item fields are present in
extracted invoice data. Missing fields trigger a 'flag' status per
config/rules.yaml validation_policies.missing_field_action.
"""

import logging

from ai_invoice_auditor.config import get_config

logger = logging.getLogger(__name__)


def data_completeness_checker_tool(
    extracted: dict, config: dict | None = None
) -> dict:
    """Check extracted invoice data for missing required fields.

    Args:
        extracted: Dict of extracted invoice fields (header + line_items).
        config: Optional rules config dict. Loaded from rules.yaml if None.

    Returns:
        Dict with keys:
            missing_fields: List of missing header-level field names.
            all_missing: List of all missing fields including line-item paths.
            status: 'flag' if any header fields missing, else 'pass'.
    """
    if config is None:
        config = get_config()

    # --- Header-level checks ---
    required_header = config["required_fields"]["header"]
    missing_fields: list[str] = []

    for field in required_header:
        value = extracted.get(field)
        if value is None:
            missing_fields.append(field)
        elif isinstance(value, str) and value.strip() == "":
            missing_fields.append(field)

    # --- Line-item checks ---
    required_line = config["required_fields"]["line_item"]
    all_missing: list[str] = list(missing_fields)

    line_items = extracted.get("line_items", [])
    for i, item in enumerate(line_items):
        for field in required_line:
            value = item.get(field)
            if value is None:
                all_missing.append(f"line_items[{i}].{field}")
            elif isinstance(value, str) and value.strip() == "":
                all_missing.append(f"line_items[{i}].{field}")

    # --- Determine status ---
    if missing_fields:
        status = config.get("validation_policies", {}).get(
            "missing_field_action", "flag"
        )
    else:
        status = "pass"

    logger.info(
        "Completeness check: %d header missing, %d total missing, status=%s",
        len(missing_fields),
        len(all_missing),
        status,
    )

    return {
        "missing_fields": missing_fields,
        "all_missing": all_missing,
        "status": status,
    }
