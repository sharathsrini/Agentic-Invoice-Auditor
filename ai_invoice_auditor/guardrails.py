"""RAI guardrails -- prompt injection sanitization, PII masking, and
confidence-gated auto-approval blocking.

Functions:
    sanitize_text(text) -- Strip prompt injection patterns from invoice text
    mask_pii(text) -- Mask phone numbers and IBANs in output text
    block_low_confidence_auto_approval(status, confidence, threshold) --
        Override auto_approve to manual_review when translation confidence is low
"""

import logging
import re

logger = logging.getLogger(__name__)

# --- Prompt Injection Patterns ---

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"system\s*:", re.IGNORECASE),
    re.compile(r"you\s+are\s+now", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?previous", re.IGNORECASE),
]


def sanitize_text(text: str) -> str:
    """Replace prompt injection patterns in text with [REDACTED].

    Args:
        text: Raw invoice text that may contain injection attempts.

    Returns:
        Sanitized text with injection patterns replaced by [REDACTED].
    """
    for pattern in INJECTION_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


# --- PII Patterns (phone + IBAN only -- see research pitfall 4) ---

PII_PATTERNS = [
    # IBAN must come first (longer match takes priority)
    (
        re.compile(
            r"[A-Z]{2}\d{2}\s?[\dA-Z]{4}\s?[\dA-Z]{4}\s?[\dA-Z]{4}\s?[\dA-Z]{0,4}\s?[\dA-Z]{0,4}\s?[\dA-Z]{0,2}"
        ),
        "[IBAN_MASKED]",
    ),
    # Phone number pattern
    (
        re.compile(
            r"\+?\(?\d{1,4}\)?[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{2,9}"
        ),
        "[PHONE_MASKED]",
    ),
]


def mask_pii(text: str) -> str:
    """Mask phone numbers and IBANs in text.

    Args:
        text: Text that may contain PII (phone numbers, IBANs).

    Returns:
        Text with phone numbers replaced by [PHONE_MASKED] and
        IBANs replaced by [IBAN_MASKED].
    """
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# --- Confidence-gated auto-approval blocking ---


def block_low_confidence_auto_approval(
    status: str,
    translation_confidence: float,
    threshold: float = 0.70,
) -> str:
    """Block auto-approval when translation confidence is below threshold.

    Defense-in-depth check: even if business validation returns auto_approve,
    this guardrail forces manual_review when the translation quality is
    uncertain.

    Args:
        status: Current pipeline status (e.g. "auto_approve", "flag", "manual_review").
        translation_confidence: Translation confidence score (0.0 to 1.0).
        threshold: Minimum confidence for auto-approval (default 0.70).

    Returns:
        Original status, or "manual_review" if auto_approve was blocked.
    """
    if status == "auto_approve" and translation_confidence < threshold:
        logger.warning(
            "Guardrail: blocking auto-approval -- translation_confidence %.2f < %.2f threshold",
            translation_confidence,
            threshold,
        )
        return "manual_review"
    return status
