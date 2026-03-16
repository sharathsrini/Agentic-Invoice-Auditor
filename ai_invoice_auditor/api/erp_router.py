"""FastAPI ERP mock router serving vendor, PO, and SKU data from JSON fixtures."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/erp", tags=["ERP Mock"])

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ERP_mockdata"


def _load_json(filename: str) -> list[dict]:
    """Load a JSON file from the ERP mock data directory.

    Args:
        filename: Name of the JSON file to load.

    Returns:
        Parsed JSON as a list of dicts.
    """
    with open(DATA_DIR / filename) as f:
        return json.load(f)


# Module-level data loading (loaded ONCE at import, NOT per-request)
_vendors = {v["vendor_id"]: v for v in _load_json("vendors.json")}
_pos = {p["po_number"]: p for p in _load_json("PO Records.json")}
_skus = {s["item_code"]: s for s in _load_json("sku_master.json")}


@router.get("/vendor/{vendor_id}")
def get_vendor(vendor_id: str) -> dict:
    """Look up a vendor by ID."""
    if vendor_id not in _vendors:
        raise HTTPException(status_code=404, detail=f"Vendor {vendor_id} not found")
    return _vendors[vendor_id]


@router.get("/po/{po_number}")
def get_po(po_number: str) -> dict:
    """Look up a purchase order by PO number."""
    if po_number not in _pos:
        raise HTTPException(status_code=404, detail=f"PO {po_number} not found")
    return _pos[po_number]


@router.get("/sku/{item_code}")
def get_sku(item_code: str) -> dict:
    """Look up a SKU by item code."""
    if item_code not in _skus:
        raise HTTPException(status_code=404, detail=f"SKU {item_code} not found")
    return _skus[item_code]
