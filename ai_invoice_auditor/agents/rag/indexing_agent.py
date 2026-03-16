"""RAG indexing agent -- chunks and stores invoice text in ChromaDB.

Runs as the final node in the 7-node pipeline graph. Handles deduplication
by deleting existing chunks for the same invoice_no before inserting new ones.
"""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_invoice_auditor.models.state import InvoiceState
from ai_invoice_auditor.agents.rag.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


def index_node(state: InvoiceState) -> dict:
    """
    LangGraph node: Chunks and indexes invoice text in ChromaDB.

    Runs as the final node in the pipeline graph with InvoiceState.
    - Extracts translated (or raw) text from state
    - Deduplicates by removing existing chunks for the same invoice_no
    - Splits text with RecursiveCharacterTextSplitter(512, 50)
    - Stores chunks in ChromaDB with invoice_no/vendor/po metadata
    """
    try:
        logger.info(f"[index_node] Processing: {state.get('file_path', 'unknown')}")

        # Extract invoice number for metadata and deduplication
        extracted = state.get("extracted", {}) or {}
        invoice_no = extracted.get("invoice_no", "unknown")

        # Prefer translated text, fall back to raw_text
        translated = state.get("translated", {}) or {}
        text = translated.get("translated_text") or state.get("raw_text", "") or ""

        if not text.strip():
            logger.warning(f"[index_node] No text to index for {invoice_no}")
            return {"messages": [f"index_node: no text to index for {invoice_no}"]}

        # Get shared vectorstore
        vectorstore = get_vectorstore()

        # Deduplication (RAG-02): delete existing chunks for this invoice
        collection = vectorstore._collection
        existing = collection.get(where={"invoice_no": invoice_no})
        if existing["ids"]:
            count = len(existing["ids"])
            collection.delete(ids=existing["ids"])
            logger.info(f"[index_node] Deleted {count} existing chunks for {invoice_no}")

        # Chunking (RAG-01): split text into overlapping chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = splitter.split_text(text)

        # Build metadata for each chunk
        metadata = {
            "invoice_no": invoice_no,
            "vendor": extracted.get("vendor_id", ""),
            "po": extracted.get("po_number", ""),
        }

        # Store chunks in ChromaDB
        vectorstore.add_texts(texts=chunks, metadatas=[metadata] * len(chunks))
        logger.info(f"[index_node] Indexed {len(chunks)} chunks for {invoice_no}")

        return {"messages": [f"index_node: indexed {len(chunks)} chunks for {invoice_no}"]}

    except FileNotFoundError as e:
        logger.error(f"[index_node] File not found: {e}")
        return {
            "status": "reject",
            "messages": [f"ERROR [index_node]: file not found -- {e}"],
        }
    except Exception as e:
        logger.error(f"[index_node] Failed: {e}")
        return {
            "status": "flag",
            "messages": [f"ERROR [index_node]: {e}"],
        }
