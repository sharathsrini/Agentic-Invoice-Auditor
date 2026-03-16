"""Shared ChromaDB vectorstore singleton for the RAG pipeline.

Provides a lazy-initialized Chroma vectorstore used by both the write path
(index_node in the pipeline graph) and the read path (retrieve_node in the
RAG query graph).
"""

import logging

from langchain_chroma import Chroma

from ai_invoice_auditor.llm import get_llm

logger = logging.getLogger(__name__)

_vectorstore = None


def get_vectorstore() -> Chroma:
    """Return the shared ChromaDB vectorstore instance (lazy singleton).

    Uses the embeddings model from the configured LLM provider.
    Collection name: 'invoice_chunks'. Data persisted to './chroma_db'.

    Returns:
        Chroma vectorstore instance.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    _, _, embeddings = get_llm()
    _vectorstore = Chroma(
        collection_name="invoice_chunks",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    logger.info("ChromaDB vectorstore initialized (collection=invoice_chunks)")
    return _vectorstore
