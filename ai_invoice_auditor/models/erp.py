"""ERP entity data models."""

from pydantic import BaseModel


class Vendor(BaseModel):
    """Vendor record from the ERP system."""

    vendor_id: str
    vendor_name: str
    country: str
    currency: str


class PurchaseOrder(BaseModel):
    """Purchase order record from the ERP system."""

    po_number: str
    vendor_id: str
    line_items: list[dict]


class SKU(BaseModel):
    """SKU (stock-keeping unit) record from the ERP system."""

    item_code: str
    category: str
    uom: str
    gst_rate: float
