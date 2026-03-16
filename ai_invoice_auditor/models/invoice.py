"""Invoice extraction data models.

Permissive Pydantic v2 models for LLM structured output -- all extraction
fields are Optional with None defaults so malformed invoices still produce
audit outcomes.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ExtractedLineItem(BaseModel):
    """A single line item extracted from an invoice."""

    item_code: Optional[str] = None
    description: Optional[str] = None
    qty: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None


class ExtractedInvoiceHeader(BaseModel):
    """Top-level invoice header fields extracted by the LLM.

    All fields are Optional to support partial extraction from malformed
    or incomplete invoices.
    """

    invoice_no: Optional[str] = None
    invoice_date: Optional[str] = None
    po_number: Optional[str] = None
    vendor_id: Optional[str] = None
    currency: Optional[str] = None
    total_amount: Optional[float] = None
    line_items: list[ExtractedLineItem] = Field(default_factory=list)


class TranslationResult(BaseModel):
    """Result of translating invoice text to English."""

    original_text: str
    translated_text: str
    source_language: str
    translation_confidence: float
    is_english: bool = False


class MissingFieldReport(BaseModel):
    """Report of a single missing required field."""

    field_name: str
    field_location: str = "header"
    severity: str = "required"
