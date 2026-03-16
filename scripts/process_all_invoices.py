#!/usr/bin/env python3
"""Process all 6 invoices through the FastAPI pipeline.

Run this after starting the server:
    uvicorn ai_invoice_auditor.main:app --reload

Then run:
    python scripts/process_all_invoices.py

After completion, open Streamlit to explore:
    streamlit run ai_invoice_auditor/streamlit_app.py
"""

import json
import sys
import time

import requests

API = "http://localhost:8000"

INVOICES = [
    {
        "file_path": "data/incoming/INV_EN_001.pdf",
        "meta": {
            "invoice_no": "INV-1001",
            "sender": "accounts@globallogistics.com",
            "subject": "Invoice INV-1001 for PO-1001",
            "language": "en",
            "attachments": ["INV_EN_001.pdf"],
        },
    },
    {
        "file_path": "data/incoming/INV_EN_002.pdf",
        "meta": {
            "invoice_no": "INV-1002",
            "sender": "finance@blueoceantransport.com",
            "subject": "Invoice INV-1002 - Container Seals Delivery",
            "language": "en",
            "attachments": ["INV_EN_002.pdf"],
        },
    },
    {
        "file_path": "data/incoming/INV_ES_003.pdf",
        "meta": {
            "invoice_no": "FAC-2025-003",
            "sender": "facturacion@transporteiberico.es",
            "subject": "Factura FAC-2025-003 - Pedido PO-1003",
            "language": "es",
            "attachments": ["INV_ES_003.pdf"],
        },
    },
    {
        "file_path": "data/incoming/INV_DE_004.pdf",
        "meta": {
            "invoice_no": "RE-2025-004",
            "sender": "rechnung@hafenlogistik.de",
            "subject": "Rechnung RE-2025-004 - Bestellung PO-1004",
            "language": "de",
            "attachments": ["INV_DE_004.pdf"],
        },
    },
    {
        "file_path": "data/incoming/INV_EN_005_scan.png",
        "meta": {
            "invoice_no": "INV-1005",
            "sender": "billing@swiftmovecouriers.com",
            "subject": "Invoice INV-1005 - Fuel and Express Box Charges",
            "language": "en",
            "attachments": ["INV_EN_005_scan.png"],
        },
    },
    {
        "file_path": "data/incoming/INV_EN_006_malformed.pdf",
        "meta": {
            "invoice_no": "INV-1006",
            "sender": "accounts@oceanfreight.in",
            "subject": "Invoice for Container Transport - PO-1006",
            "language": "en",
            "attachments": ["INV_EN_006_malformed.pdf"],
        },
    },
]


def main():
    # Health check
    try:
        r = requests.get(f"{API}/health", timeout=5)
        r.raise_for_status()
    except requests.ConnectionError:
        print("ERROR: Cannot connect to API. Start the server first:")
        print("  uvicorn ai_invoice_auditor.main:app --reload")
        sys.exit(1)

    print("=" * 70)
    print("  AI Invoice Auditor — Processing All 6 Invoices")
    print("=" * 70)

    results = []

    for inv in INVOICES:
        invoice_no = inv["meta"]["invoice_no"]
        lang = inv["meta"]["language"]
        print(f"\n--- {invoice_no} ({lang}, {inv['file_path']}) ---")

        start = time.time()
        resp = requests.post(f"{API}/invoice/process", json=inv, timeout=120)
        elapsed = time.time() - start
        data = resp.json()

        status = data.get("status", "unknown")
        report = data.get("report_path", "")
        messages = data.get("messages", [])

        # Leave interrupted invoices for Streamlit Human Review tab
        if status == "interrupted":
            print(f"  >>> NEEDS HUMAN REVIEW — open Streamlit to approve/reject/edit")
            print(f"  >>> Or resume via API: curl -X POST {API}/invoice/{invoice_no}/resume -H 'Content-Type: application/json' -d '{{\"decision\": \"approve\"}}'")
            status = "interrupted (awaiting review)"

        print(f"  Status: {status}")
        print(f"  Report: {report or 'N/A'}")
        print(f"  Time:   {elapsed:.1f}s")
        print(f"  Nodes:  {len(messages)}")

        results.append({
            "invoice_no": invoice_no,
            "language": lang,
            "status": status,
            "report": report,
            "time": round(elapsed, 1),
            "nodes": len(messages),
        })

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Invoice':<16} {'Lang':<5} {'Status':<30} {'Time':>6} {'Nodes':>5}")
    print(f"  {'-'*16} {'-'*5} {'-'*30} {'-'*6} {'-'*5}")
    for r in results:
        print(f"  {r['invoice_no']:<16} {r['language']:<5} {r['status']:<30} {r['time']:>5.1f}s {r['nodes']:>5}")

    # List all processed
    print(f"\n--- Invoice List (GET /invoice/) ---")
    list_resp = requests.get(f"{API}/invoice/", timeout=5)
    if list_resp.status_code == 200:
        for inv in list_resp.json().get("invoices", []):
            print(f"  {inv['invoice_no']}: {inv['status']}")

    # Quick RAG test
    print(f"\n--- RAG Test ---")
    rag_resp = requests.post(
        f"{API}/rag/query",
        json={"query": "What is the total amount on INV-1001?"},
        timeout=60,
    )
    if rag_resp.status_code == 200:
        rd = rag_resp.json()
        print(f"  Q: What is the total amount on INV-1001?")
        print(f"  A: {rd.get('answer', 'N/A')[:200]}")
        print(f"  Triad: {rd.get('triad_scores', {})}")

    print(f"\n{'=' * 70}")
    print("  Done! Open Streamlit to explore:")
    print("    streamlit run ai_invoice_auditor/streamlit_app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
