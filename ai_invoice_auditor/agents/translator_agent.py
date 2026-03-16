import logging

from ai_invoice_auditor.models.state import InvoiceState
from ai_invoice_auditor.tools.lang_bridge import lang_bridge_tool

logger = logging.getLogger(__name__)


def translate_node(state: InvoiceState) -> dict:
    """
    LangGraph node: Translates non-English invoice text to English.

    - English invoices bypass LLM entirely (confidence 1.0).
    - Non-English invoices are translated via lang_bridge_tool.
    - Translations with confidence < 0.70 route to manual_review.
    """
    try:
        file_path = state.get("file_path", "unknown")
        logger.info("[translate_node] Processing: %s", file_path)

        meta = state.get("meta", {})
        language = meta.get("language", "en")
        raw_text = state.get("raw_text", "")

        # English bypass -- no LLM call needed
        if language == "en" or language.startswith("en"):
            logger.info(
                "[translate_node] English detected (lang=%s), skipping translation",
                language,
            )
            return {
                "translated": {
                    "original_text": raw_text,
                    "translated_text": raw_text,
                    "source_language": "en",
                    "translation_confidence": 1.0,
                    "is_english": True,
                },
                "messages": [
                    "translate_node: English detected, skipping translation"
                ],
            }

        # Non-English: call lang_bridge_tool for LLM translation
        logger.info(
            "[translate_node] Translating from '%s' via lang_bridge_tool", language
        )
        translation_result = lang_bridge_tool(raw_text, language)

        # Confidence routing -- low confidence goes to manual_review
        confidence = translation_result.get("translation_confidence", 0.0)
        if confidence < 0.70:
            logger.info(
                "[translate_node] Low confidence (%.2f), routing to manual_review",
                confidence,
            )
            return {
                "translated": translation_result,
                "status": "manual_review",
                "messages": [
                    f"translate_node: low translation confidence ({confidence:.2f}), "
                    "routing to manual_review"
                ],
            }

        logger.info(
            "[translate_node] Translation complete (confidence=%.2f)", confidence
        )
        return {
            "translated": translation_result,
            "messages": [
                f"translate_node: translated from '{language}' "
                f"(confidence={confidence:.2f})"
            ],
        }

    except FileNotFoundError as e:
        logger.error("[translate_node] File not found: %s", e)
        return {
            "status": "reject",
            "messages": [f"ERROR [translate_node]: file not found -- {e}"],
        }
    except Exception as e:
        logger.error("[translate_node] Failed: %s", e)
        return {
            "status": "flag",
            "messages": [f"ERROR [translate_node]: {e}"],
        }
