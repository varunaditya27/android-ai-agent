"""
OCR Utilities
=============

Text recognition utilities for extracting text from screenshots.
Used when accessibility tree text is unavailable or incomplete.

Note: Primary text extraction should come from the accessibility tree.
OCR is a fallback for custom views or when tree parsing fails.
"""

import re
from typing import Any, Optional

from app.llm.client import LLMClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def extract_text_with_vision(
    llm_client: LLMClient,
    screenshot_b64: str,
    region: Optional[dict[str, int]] = None,
) -> str:
    """
    Extract text from screenshot using vision LLM.

    Args:
        llm_client: LLM client with vision support.
        screenshot_b64: Base64-encoded screenshot.
        region: Optional region to focus on (left, top, right, bottom).

    Returns:
        Extracted text content.
    """
    prompt = """Extract all visible text from this Android screenshot.

Return the text in reading order (top to bottom, left to right).
Include:
- Button labels
- Menu items
- Input field text and placeholders
- Status bar text
- Any other visible text

Format as plain text, one element per line.
Do not add commentary or descriptions."""

    if region:
        prompt += f"\n\nFocus on the region: x={region.get('left', 0)}-{region.get('right', 0)}, y={region.get('top', 0)}-{region.get('bottom', 0)}"

    try:
        response = await llm_client.complete_with_vision(
            prompt=prompt,
            image_data=screenshot_b64,
            system_prompt="You are an OCR system. Extract text accurately.",
        )
        return response.content.strip()
    except Exception as e:
        logger.warning("Vision OCR failed", error=str(e))
        return ""


async def find_text_location(
    llm_client: LLMClient,
    screenshot_b64: str,
    target_text: str,
) -> Optional[dict[str, int]]:
    """
    Find the location of specific text in a screenshot.

    Args:
        llm_client: LLM client with vision support.
        screenshot_b64: Base64-encoded screenshot.
        target_text: Text to locate.

    Returns:
        Bounding box dict or None if not found.
    """
    prompt = f"""Find the text "{target_text}" in this Android screenshot.

If found, return the approximate bounding box as percentages:
{{"found": true, "bounds": [left%, top%, right%, bottom%]}}

If not found:
{{"found": false}}

Only return the JSON, no other text."""

    try:
        response = await llm_client.complete_with_vision(
            prompt=prompt,
            image_data=screenshot_b64,
        )

        # Parse JSON response
        import json

        match = re.search(r"\{[^}]+\}", response.content)
        if match:
            data = json.loads(match.group())
            if data.get("found") and "bounds" in data:
                bounds_pct = data["bounds"]
                # Convert percentages to approximate pixels (assuming 1080x2340)
                return {
                    "left": int(bounds_pct[0] * 1080 / 100),
                    "top": int(bounds_pct[1] * 2340 / 100),
                    "right": int(bounds_pct[2] * 1080 / 100),
                    "bottom": int(bounds_pct[3] * 2340 / 100),
                }
    except Exception as e:
        logger.warning("Text location failed", error=str(e), target=target_text)

    return None


def extract_numbers(text: str) -> list[str]:
    """
    Extract numbers from text.

    Useful for extracting prices, quantities, codes, etc.

    Args:
        text: Text to extract from.

    Returns:
        List of number strings found.
    """
    # Match integers, decimals, and formatted numbers
    pattern = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"
    return re.findall(pattern, text)


def extract_phone_numbers(text: str) -> list[str]:
    """
    Extract phone numbers from text.

    Args:
        text: Text to search.

    Returns:
        List of phone number strings.
    """
    # Various phone number formats
    patterns = [
        r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        r"\d{10}",
        r"\d{3}[-.\s]\d{3}[-.\s]\d{4}",
    ]

    numbers = []
    for pattern in patterns:
        numbers.extend(re.findall(pattern, text))

    return list(set(numbers))


def extract_emails(text: str) -> list[str]:
    """
    Extract email addresses from text.

    Args:
        text: Text to search.

    Returns:
        List of email addresses.
    """
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return re.findall(pattern, text)


def extract_urls(text: str) -> list[str]:
    """
    Extract URLs from text.

    Args:
        text: Text to search.

    Returns:
        List of URLs.
    """
    pattern = r"https?://[^\s<>\"{}|\\^`\[\]]+"
    return re.findall(pattern, text)


def clean_ocr_text(text: str) -> str:
    """
    Clean and normalize OCR output.

    Args:
        text: Raw OCR text.

    Returns:
        Cleaned text.
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove common OCR artifacts
    text = re.sub(r"[|]", "l", text)  # Pipe often misread as l
    text = re.sub(r"[0O]", lambda m: "O" if m.group().isupper() else "0", text)

    return text.strip()
