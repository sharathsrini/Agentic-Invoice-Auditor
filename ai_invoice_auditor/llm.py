"""Dual-provider LLM initialization module.

Supports Azure OpenAI and Google Gemini with lazy initialization
and module-level caching. Provider selection is controlled by the
MODEL_PROVIDER env var ("auto", "azure", "gemini") with auto as default.
"""

import os
import logging

logger = logging.getLogger(__name__)

_provider: str | None = None
_chat_model = None
_embeddings = None


def get_llm():
    """Return (provider_name, chat_model, embeddings). Lazy-initialized, cached.

    Provider selection order:
      1. If MODEL_PROVIDER is set to "azure" or "gemini", use that provider.
      2. If MODEL_PROVIDER is "auto" (default), try Azure first, then Gemini.
      3. Azure requires all 4 env vars: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
         AZURE_OPENAI_CHAT_DEPLOYMENT, AZURE_OPENAI_EMBEDDING_DEPLOYMENT.
      4. Gemini requires GEMINI_API_KEY.

    Returns:
        Tuple of (provider_name: str, chat_model, embeddings).

    Raises:
        RuntimeError: If no LLM provider is configured.
    """
    global _provider, _chat_model, _embeddings
    if _chat_model is not None:
        return _provider, _chat_model, _embeddings

    mode = os.getenv("MODEL_PROVIDER", "auto")

    # XCUT-07: Azure enforcement -- if all 4 Azure env vars are present,
    # force Azure regardless of MODEL_PROVIDER setting
    azure_env_vars = [
        os.getenv("AZURE_OPENAI_API_KEY"),
        os.getenv("AZURE_OPENAI_ENDPOINT"),
        os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    ]
    if all(azure_env_vars):
        if mode != "azure":
            logger.info("Provider enforcement: Azure credentials detected, using Azure")
        mode = "azure"

    # Try Azure first
    if mode in ("auto", "azure"):
        azure_ready = all([
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        ])
        if azure_ready:
            from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
            _provider = "azure"
            _chat_model = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
                temperature=0,
            )
            _embeddings = AzureOpenAIEmbeddings(
                deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            )
            logger.info("LLM provider: Azure OpenAI")
            return _provider, _chat_model, _embeddings

    # Try Gemini
    if mode in ("auto", "gemini") and os.getenv("GEMINI_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        _provider = "gemini"
        _chat_model = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
        )
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=f"models/{os.getenv('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001')}",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
        logger.info("LLM provider: Gemini")
        return _provider, _chat_model, _embeddings

    raise RuntimeError(
        "No LLM provider configured. Set AZURE_OPENAI_* or GEMINI_API_KEY env vars."
    )
