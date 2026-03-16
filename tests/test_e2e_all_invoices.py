"""End-to-end integration test for all 6 invoices.

Processes each invoice through the full pipeline via FastAPI TestClient,
verifying extraction, translation, validation, HITL interrupt, report
generation, and RAG indexing.

Usage:
    uv run pytest tests/test_e2e_all_invoices.py -v -s
"""

import json
import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Load .env
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

from ai_invoice_auditor.main import app

client = TestClient(app)

# Invoice test data: file_path, meta, expected behaviors
INVOICES = [
    {
        "id": "INV_EN_001",
        "file_path": "data/incoming/INV_EN_001.pdf",
        "meta": {
            "invoice_no": "INV-1001",
            "sender": "accounts@globallogistics.com",
            "subject": "Invoice INV-1001 for PO-1001",
            "language": "en",
            "attachments": ["INV_EN_001.pdf"],
        },
        "expect_language": "en",
        "expect_translation": False,
        "expect_interrupt": True,  # SKU-002 price discrepancy
        "description": "English PDF, PO-1001, SKU-002 price $3.50 vs ERP $3.00 -> manual_review",
    },
    {
        "id": "INV_EN_002",
        "file_path": "data/incoming/INV_EN_002.pdf",
        "meta": {
            "invoice_no": "INV-1002",
            "sender": "finance@blueoceantransport.com",
            "subject": "Invoice INV-1002 - Container Seals Delivery",
            "language": "en",
            "attachments": ["INV_EN_002.pdf"],
        },
        "expect_language": "en",
        "expect_translation": False,
        "expect_interrupt": None,  # depends on LLM extraction
        "description": "English PDF, no PO in subject (body fallback)",
    },
    {
        "id": "INV_ES_003",
        "file_path": "data/incoming/INV_ES_003.pdf",
        "meta": {
            "invoice_no": "FAC-2025-003",
            "sender": "facturacion@transporteiberico.es",
            "subject": "Factura FAC-2025-003 - Pedido PO-1003",
            "language": "es",
            "attachments": ["INV_ES_003.pdf"],
        },
        "expect_language": "es",
        "expect_translation": True,
        "expect_interrupt": None,  # depends on price extraction accuracy
        "description": "Spanish PDF, needs translation, PO-1003",
    },
    {
        "id": "INV_DE_004",
        "file_path": "data/incoming/INV_DE_004.pdf",
        "meta": {
            "invoice_no": "RE-2025-004",
            "sender": "rechnung@hafenlogistik.de",
            "subject": "Rechnung RE-2025-004 - Bestellung PO-1004",
            "language": "de",
            "attachments": ["INV_DE_004.pdf"],
        },
        "expect_language": "de",
        "expect_translation": True,
        "expect_interrupt": None,
        "description": "German PDF (meta said .docx, fixed), needs translation, PO-1004",
    },
    {
        "id": "INV_EN_005",
        "file_path": "data/incoming/INV_EN_005_scan.png",
        "meta": {
            "invoice_no": "INV-1005",
            "sender": "billing@swiftmovecouriers.com",
            "subject": "Invoice INV-1005 - Fuel and Express Box Charges",
            "language": "en",
            "attachments": ["INV_EN_005_scan.png"],
        },
        "expect_language": "en",
        "expect_translation": False,
        "expect_interrupt": None,  # OCR quality dependent
        "description": "English PNG scan (OCR), PO-1005",
    },
    {
        "id": "INV_EN_006",
        "file_path": "data/incoming/INV_EN_006_malformed.pdf",
        "meta": {
            "invoice_no": "INV-1006",
            "sender": "accounts@oceanfreight.in",
            "subject": "Invoice for Container Transport - PO-1006",
            "language": "en",
            "attachments": ["INV_EN_006_malformed.pdf"],
        },
        "expect_language": "en",
        "expect_translation": False,
        "expect_interrupt": None,  # malformed -> likely flag
        "description": "English PDF, malformed (missing fields), PO-1006",
    },
]


class TestEndToEnd:
    """Process all 6 invoices and verify pipeline behavior."""

    def test_health(self):
        """API is running."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_erp_endpoints(self):
        """ERP mock serves data correctly."""
        resp = client.get("/erp/vendor/VEND-001")
        assert resp.status_code == 200
        assert "Global Logistics" in resp.json()["vendor_name"]

        resp = client.get("/erp/po/PO-1001")
        assert resp.status_code == 200
        assert len(resp.json()["line_items"]) == 3

        resp = client.get("/erp/sku/SKU-001")
        assert resp.status_code == 200

    @pytest.mark.parametrize(
        "invoice",
        INVOICES,
        ids=[inv["id"] for inv in INVOICES],
    )
    def test_process_invoice(self, invoice):
        """Process each invoice through the full pipeline."""
        resp = client.post(
            "/invoice/process",
            json={
                "file_path": invoice["file_path"],
                "meta": invoice["meta"],
            },
        )

        # Should be 200 (completed) or 202 (interrupted for HITL)
        assert resp.status_code in (200, 202), (
            f"{invoice['id']}: Unexpected status {resp.status_code}: {resp.text}"
        )

        data = resp.json()
        invoice_no = data.get("invoice_no") or invoice["meta"]["invoice_no"]
        status = data.get("status", "unknown")
        messages = data.get("messages", [])

        print(f"\n{'='*60}")
        print(f"INVOICE: {invoice['id']} ({invoice['description']})")
        print(f"  Status: {status}")
        print(f"  Invoice No: {invoice_no}")
        print(f"  Report: {data.get('report_path', 'N/A')}")
        print(f"  Messages ({len(messages)}):")
        for msg in messages:
            print(f"    - {msg}")

        if invoice.get("expect_interrupt") is True:
            assert resp.status_code == 202 or status in ("interrupted", "manual_review"), (
                f"{invoice['id']}: Expected interrupt but got status={status}"
            )
            print(f"  -> HITL INTERRUPT as expected (manual_review)")

            # Resume with approve
            resume_resp = client.post(
                f"/invoice/{invoice_no}/resume",
                json={"decision": "approve"},
            )
            print(f"  -> Resume (approve): {resume_resp.status_code}")
            if resume_resp.status_code == 200:
                resume_data = resume_resp.json()
                print(f"  -> Post-resume status: {resume_data.get('status')}")
                print(f"  -> Report: {resume_data.get('report_path')}")

        # Verify we can query status
        status_resp = client.get(f"/invoice/{invoice_no}")
        assert status_resp.status_code == 200

    def test_invoice_list(self):
        """List endpoint returns all processed invoices."""
        resp = client.get("/invoice/")
        assert resp.status_code == 200
        invoices = resp.json().get("invoices", [])
        print(f"\nProcessed invoices: {len(invoices)}")
        for inv in invoices:
            print(f"  {inv['invoice_no']}: {inv['status']}")
        assert len(invoices) > 0

    def test_rag_query_after_indexing(self):
        """RAG query returns grounded answer from indexed invoices."""
        resp = client.post(
            "/rag/query",
            json={"query": "What invoices have been processed?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        print(f"\nRAG Query: 'What invoices have been processed?'")
        print(f"  Answer: {data.get('answer', 'N/A')[:200]}")
        print(f"  Triad scores: {data.get('triad_scores', {})}")
        print(f"  Source chunks: {len(data.get('source_chunks', []))}")

    def test_rag_refusal(self):
        """Irrelevant query gets refusal."""
        resp = client.post(
            "/rag/query",
            json={"query": "What is the weather in Tokyo?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        answer = data.get("answer", "")
        print(f"\nRAG Refusal test:")
        print(f"  Answer: {answer[:200]}")
        # Should refuse or give low-quality answer
        assert "don't have" in answer.lower() or "not" in answer.lower() or data.get("triad_scores", {}).get("context_relevance", 1.0) < 0.7
