"""Lang-Bridge Tool -- translates non-English invoice text to English.

Uses the active LLM provider (Azure or Gemini) with structured output
to produce translated text and a self-reported confidence score.
English text bypasses the LLM entirely.
"""

import logging

logger = logging.getLogger(__name__)


def lang_bridge_tool(text: str, source_language: str) -> dict:
    """Translate non-English invoice text to English using active LLM.

    Args:
        text: Raw invoice text in the source language.
        source_language: ISO 639-1 language code (e.g., "es", "de", "en").

    Returns:
        Dict with keys: original_text, translated_text, source_language,
        translation_confidence (float 0.0-1.0), and is_english (bool).
    """
    if source_language == "en":
        return {
            "original_text": text,
            "translated_text": text,
            "source_language": "en",
            "translation_confidence": 1.0,
            "is_english": True,
        }

    from ai_invoice_auditor.llm import get_llm
    from ai_invoice_auditor.models.invoice import TranslationResult

    _, chat_model, _ = get_llm()

    # Use structured output for reliable JSON response
    translator = chat_model.with_structured_output(TranslationResult)

    prompt = (
        f"Translate the following {source_language} invoice text to English.\n"
        "Preserve all numbers, dates, currency values, and invoice field labels exactly.\n"
        "Report your translation_confidence as a float between 0.0 and 1.0:\n"
        "- 1.0 = perfect confidence (common business terms, clear text)\n"
        "- 0.90-0.99 = high confidence (some domain-specific terms)\n"
        "- 0.70-0.89 = moderate confidence (ambiguous terms, unclear context)\n"
        "- Below 0.70 = low confidence (many uncertain translations)\n"
        "If you encounter technical jargon, abbreviations, or domain-specific terms\n"
        "that could have multiple valid translations, report confidence below 0.95.\n"
        "\n"
        f"Source language: {source_language}\n"
        f"Text to translate:\n{text}"
    )

    logger.info(
        "Translating %d chars from '%s' to English", len(text), source_language
    )
    result = translator.invoke(prompt)
    output = result.model_dump()
    # Override is_english: we know the source isn't English (we already
    # bypassed above for source_language=="en"). The LLM may set
    # is_english=True because the *translated* output is English.
    output["is_english"] = False
    output["source_language"] = source_language
    return output
