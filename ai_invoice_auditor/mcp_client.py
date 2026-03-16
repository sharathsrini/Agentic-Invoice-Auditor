"""MCP client helper -- try-MCP/except-direct-import pattern for agent nodes.

Provides call_tool() as the primary MCP-based tool invocation path per ORCH-06.
If the MCP call fails for any reason (import error, event loop conflict, server
not running), it silently falls back to a user-supplied fallback function.

Usage:
    from ai_invoice_auditor.mcp_client import call_tool

    result = call_tool("completeness_checker", {"extracted": data},
                       fallback_fn=data_completeness_checker_tool)
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


def call_tool(tool_name: str, args: dict, fallback_fn=None):
    """Invoke an MCP tool by name, falling back to direct import on failure.

    Tries the MCP path first (in-process FastMCP Client). On ANY exception,
    falls back to calling fallback_fn(**args) if provided.

    Args:
        tool_name: Name of the registered MCP tool to call.
        args: Dict of keyword arguments to pass to the tool.
        fallback_fn: Optional callable to use when MCP is unavailable.
            Called as fallback_fn(**args).

    Returns:
        Tool result from MCP or fallback.

    Raises:
        Exception: Re-raises the MCP exception if no fallback_fn is provided.
    """
    try:
        from fastmcp import Client
        from ai_invoice_auditor.mcp_server import mcp as server

        async def _call():
            async with Client(server) as client:
                return await client.call_tool(tool_name, args)

        # Handle event loop: if one is already running, we cannot use
        # asyncio.run(). Fall through to fallback in that case.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # An event loop is already running (e.g., inside Jupyter or
            # an async framework). Cannot use asyncio.run() here.
            raise RuntimeError(
                "Event loop already running; falling back to direct import"
            )

        result = asyncio.run(_call())
        # FastMCP Client.call_tool returns a CallToolResult object.
        # Extract the underlying data for a clean return value.
        return _unwrap_result(result)

    except Exception as exc:
        logger.debug(
            "MCP call for %s failed (%s), using direct import fallback",
            tool_name,
            exc,
        )
        if fallback_fn is not None:
            return fallback_fn(**args)
        raise


def _unwrap_result(result):
    """Extract clean Python data from a FastMCP CallToolResult.

    FastMCP 2.x returns a CallToolResult with .data/.structured_content
    attributes. This helper extracts the underlying dict/list/str so
    callers get plain Python objects.
    """
    import json

    # FastMCP 2.x CallToolResult: prefer .data, then .structured_content
    if hasattr(result, "data") and result.data is not None:
        return result.data
    if hasattr(result, "structured_content") and result.structured_content is not None:
        return result.structured_content

    # FastMCP may return a list of content items (older versions)
    if isinstance(result, list) and len(result) == 1:
        item = result[0]
        if hasattr(item, "text"):
            try:
                return json.loads(item.text)
            except (json.JSONDecodeError, TypeError):
                return item.text

    # If result has .content with text items, try to extract
    if hasattr(result, "content") and result.content:
        texts = []
        for item in result.content:
            if hasattr(item, "text"):
                texts.append(item.text)
        if len(texts) == 1:
            try:
                return json.loads(texts[0])
            except (json.JSONDecodeError, TypeError):
                return texts[0]

    return result


def call_completeness_checker(extracted: dict) -> dict:
    """Convenience wrapper: invoke completeness_checker via MCP with fallback."""
    from ai_invoice_auditor.tools.completeness_checker import (
        data_completeness_checker_tool,
    )

    return call_tool(
        "completeness_checker",
        {"extracted": extracted},
        fallback_fn=data_completeness_checker_tool,
    )


def call_business_validator(extracted: dict) -> dict:
    """Convenience wrapper: invoke business_validator via MCP with fallback."""
    from ai_invoice_auditor.tools.business_validator import (
        business_validation_tool,
    )

    return call_tool(
        "business_validator",
        {"extracted": extracted},
        fallback_fn=business_validation_tool,
    )
