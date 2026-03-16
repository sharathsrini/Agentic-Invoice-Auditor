"""LangFuse observability integration with graceful degradation.

Provides a singleton LangFuse CallbackHandler for LangChain/LangGraph
tracing. When LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are not set,
returns None so the app continues without observability.
"""

import logging
import os

logger = logging.getLogger(__name__)

_handler = None
_initialized = False


def get_langfuse_handler():
    """Return a singleton LangFuse CallbackHandler, or None if unconfigured.

    Checks LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY env vars.
    The CallbackHandler constructor reads all env vars automatically
    (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST).

    Returns:
        CallbackHandler instance or None if credentials are missing.
    """
    global _handler, _initialized
    if _initialized:
        return _handler

    _initialized = True

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        logger.warning(
            "LangFuse credentials not found (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY). "
            "Observability disabled."
        )
        return None

    try:
        from langfuse.langchain import CallbackHandler

        _handler = CallbackHandler()
        logger.info("LangFuse CallbackHandler initialized (host=%s)", os.getenv("LANGFUSE_HOST", "default"))
        return _handler
    except ImportError:
        logger.warning("langfuse package not installed. Observability disabled.")
        return None
    except Exception as e:
        logger.warning("LangFuse initialization failed: %s. Observability disabled.", e)
        return None
