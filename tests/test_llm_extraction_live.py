"""Live integration tests for LLM provider and extraction pipeline.

Validates Gemini LLM initialization, PDF/PNG text extraction, LLM-based
translation, and the full extract_node pipeline against real invoice data.

Usage:
    GEMINI_API_KEY=<key> python tests/test_llm_extraction_live.py
"""

import logging
import os
import sys
import traceback

# Load .env file if present (for GEMINI_API_KEY etc.)
from pathlib import Path
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS: list[tuple[str, bool, str]] = []


def run_test(name: str, fn):
    """Run *fn*, capture pass/fail, append to RESULTS."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        passed, msg = fn()
    except Exception as exc:
        passed = False
        msg = f"EXCEPTION: {exc}\n{traceback.format_exc()}"
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {msg}")
    RESULTS.append((name, passed, msg))


# ---------------------------------------------------------------------------
# Test 1: LLM Provider Initialization
# ---------------------------------------------------------------------------

def test_llm_provider():
    os.environ.setdefault("MODEL_PROVIDER", "gemini")

    from ai_invoice_auditor.llm import get_llm
    provider, chat_model, embeddings = get_llm()

    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    assert provider == "gemini", f"Expected provider 'gemini', got '{provider}'"
    assert chat_model is not None, "chat_model is None"
    assert isinstance(chat_model, ChatGoogleGenerativeAI), (
        f"chat_model type: {type(chat_model).__name__}"
    )
    assert embeddings is not None, "embeddings is None"
    assert isinstance(embeddings, GoogleGenerativeAIEmbeddings), (
        f"embeddings type: {type(embeddings).__name__}"
    )
    return True, (
        f"provider={provider}, "
        f"chat_model={type(chat_model).__name__}, "
        f"embeddings={type(embeddings).__name__}"
    )


# ---------------------------------------------------------------------------
# Test 2: Data Harvester -- PDF and PNG extraction
# ---------------------------------------------------------------------------

def test_data_harvester():
    from ai_invoice_auditor.tools.data_harvester import data_harvester_tool

    # PDF extraction
    pdf_text = data_harvester_tool("data/incoming/INV_EN_001.pdf")
    assert len(pdf_text) > 50, f"PDF text too short ({len(pdf_text)} chars)"
    assert "invoice" in pdf_text.lower(), "PDF text does not contain 'invoice'"
    print(f"  PDF text snippet (first 200 chars):\n    {pdf_text[:200]!r}")

    # PNG OCR extraction
    png_text = data_harvester_tool("data/incoming/INV_EN_005_scan.png")
    assert len(png_text) > 20, f"PNG text too short ({len(png_text)} chars)"
    print(f"  PNG text snippet (first 200 chars):\n    {png_text[:200]!r}")

    return True, (
        f"PDF: {len(pdf_text)} chars extracted, "
        f"PNG: {len(png_text)} chars extracted"
    )


# ---------------------------------------------------------------------------
# Test 3: Lang Bridge -- Translation via LLM
# ---------------------------------------------------------------------------

def test_lang_bridge():
    from ai_invoice_auditor.tools.data_harvester import data_harvester_tool
    from ai_invoice_auditor.tools.lang_bridge import lang_bridge_tool

    # English bypass (no LLM call)
    en_result = lang_bridge_tool("Hello world", "en")
    assert en_result["is_english"] is True, f"Expected is_english=True, got {en_result['is_english']}"
    assert en_result["translation_confidence"] == 1.0, (
        f"Expected confidence 1.0, got {en_result['translation_confidence']}"
    )
    assert en_result["translated_text"] == "Hello world", (
        f"Expected 'Hello world', got {en_result['translated_text']!r}"
    )
    print(f"  English bypass: OK (confidence={en_result['translation_confidence']})")

    # Real translation: Spanish invoice
    spanish_text = data_harvester_tool("data/incoming/INV_ES_003.pdf")
    es_result = lang_bridge_tool(spanish_text, "es")
    assert es_result["is_english"] is False, f"Expected is_english=False, got {es_result['is_english']}"
    assert es_result["translation_confidence"] > 0.0, (
        f"Expected confidence > 0.0, got {es_result['translation_confidence']}"
    )
    assert len(es_result["translated_text"]) > 0, "Translated text is empty"
    assert es_result["translated_text"] != spanish_text, "Translated text is identical to source"
    assert es_result["source_language"] == "es", (
        f"Expected source_language='es', got {es_result['source_language']!r}"
    )
    print(f"  Spanish translation: confidence={es_result['translation_confidence']}")
    print(f"  Translated text snippet (first 200 chars):\n    {es_result['translated_text'][:200]!r}")

    return True, (
        f"English bypass OK, "
        f"Spanish translation confidence={es_result['translation_confidence']:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 4: Extractor Agent -- Full extract_node pipeline
# ---------------------------------------------------------------------------

def test_extract_node():
    from ai_invoice_auditor.agents.extractor_agent import extract_node

    state = {
        "file_path": "data/incoming/INV_EN_001.pdf",
        "meta": {
            "sender": "accounts@globallogistics.com",
            "subject": "Invoice INV-1001 for PO-1001",
            "received_timestamp": "2025-03-14T09:32:00Z",
            "language": "en",
            "attachments": ["INV_EN_001.pdf"],
        },
        "raw_text": None,
        "extracted": None,
        "translated": None,
        "validation_result": None,
        "report_path": None,
        "messages": [],
        "status": "new",
        "next": "",
    }

    result = extract_node(state)

    assert "raw_text" in result, "Result missing 'raw_text'"
    assert result["raw_text"] is not None and len(result["raw_text"]) > 0, (
        "raw_text is empty"
    )
    assert "extracted" in result, "Result missing 'extracted'"
    assert isinstance(result["extracted"], dict), (
        f"extracted is not a dict: {type(result['extracted'])}"
    )

    extracted = result["extracted"]
    key_fields = ["invoice_no", "vendor_id", "total_amount", "po_number"]
    populated = [f for f in key_fields if extracted.get(f) is not None]
    assert len(populated) > 0, (
        f"No key fields populated. extracted={extracted}"
    )

    print(f"  Populated key fields: {populated}")
    print(f"  Full extracted dict:")
    for k, v in extracted.items():
        print(f"    {k}: {v!r}")

    return True, (
        f"Populated fields: {populated}, "
        f"invoice_no={extracted.get('invoice_no')}, "
        f"total_amount={extracted.get('total_amount')}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not set. Cannot run live integration tests.")
        sys.exit(1)

    run_test("1. LLM Provider Initialization", test_llm_provider)
    run_test("2. Data Harvester (PDF + PNG)", test_data_harvester)
    run_test("3. Lang Bridge (Translation via LLM)", test_lang_bridge)
    run_test("4. Extractor Agent (extract_node)", test_extract_node)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed_count = sum(1 for _, p, _ in RESULTS if p)
    total = len(RESULTS)
    for name, passed, msg in RESULTS:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {msg}")
    print(f"\n  {passed_count}/{total} tests passed.")
    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    main()
