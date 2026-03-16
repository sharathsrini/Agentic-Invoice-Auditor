"""Validation result data models."""

from typing import Literal, Optional

from pydantic import BaseModel


class Discrepancy(BaseModel):
    """A single discrepancy found during business validation."""

    item_code: str
    field: str
    invoice_value: float | str
    erp_value: float | str
    deviation_pct: Optional[float] = None
    breaches_tolerance: bool


class ValidationResult(BaseModel):
    """Final validation outcome for an invoice."""

    invoice_no: str
    status: Literal["auto_approve", "flag", "manual_review", "reject"]
    missing_fields: list[dict] = []
    discrepancies: list[dict] = []
    translation_confidence: float = 1.0
    recommendation: str
    reviewer_decision: Optional[str] = None
