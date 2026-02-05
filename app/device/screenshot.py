"""
Screenshot Utilities
====================

Screenshot capture, processing, and optimization utilities.

Provides functions for:
- Screenshot compression and quality adjustment
- Image resizing for LLM input
- Base64 encoding/decoding
- Image annotation for debugging
"""

import base64
import io
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from app.utils.logger import get_logger

logger = get_logger(__name__)


def compress_screenshot(
    screenshot_b64: str,
    quality: int = 85,
    max_size: Optional[Tuple[int, int]] = None,
) -> str:
    """
    Compress and optionally resize a screenshot.

    Args:
        screenshot_b64: Base64-encoded PNG screenshot.
        quality: JPEG quality (1-100).
        max_size: Optional max dimensions (width, height).

    Returns:
        Base64-encoded compressed image.
    """
    # Decode base64 to image
    image_bytes = base64.b64decode(screenshot_b64)
    image = Image.open(io.BytesIO(image_bytes))

    # Resize if needed
    if max_size:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        logger.debug(
            "Screenshot resized",
            original_size=f"{image.width}x{image.height}",
            max_size=max_size,
        )

    # Convert to RGB (JPEG doesn't support alpha)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    # Compress to JPEG
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    compressed_bytes = output.getvalue()

    original_size = len(image_bytes)
    compressed_size = len(compressed_bytes)
    ratio = compressed_size / original_size * 100

    logger.debug(
        "Screenshot compressed",
        original_kb=original_size // 1024,
        compressed_kb=compressed_size // 1024,
        ratio=f"{ratio:.1f}%",
    )

    return base64.b64encode(compressed_bytes).decode("utf-8")


def resize_for_llm(
    screenshot_b64: str,
    max_width: int = 1024,
    max_height: int = 1024,
) -> str:
    """
    Resize screenshot to fit LLM input size limits.

    Most vision LLMs work best with images around 1024x1024.
    This maintains aspect ratio while fitting within limits.

    Args:
        screenshot_b64: Base64-encoded screenshot.
        max_width: Maximum width in pixels.
        max_height: Maximum height in pixels.

    Returns:
        Base64-encoded resized PNG image.
    """
    image_bytes = base64.b64decode(screenshot_b64)
    image = Image.open(io.BytesIO(image_bytes))

    # Calculate new size maintaining aspect ratio
    original_width, original_height = image.size
    ratio = min(max_width / original_width, max_height / original_height)

    if ratio < 1:
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.debug(
            "Screenshot resized for LLM",
            original=f"{original_width}x{original_height}",
            new=f"{new_width}x{new_height}",
        )

    # Save as PNG (lossless)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)

    return base64.b64encode(output.getvalue()).decode("utf-8")


def annotate_screenshot(
    screenshot_b64: str,
    elements: list[dict],
    highlight_index: Optional[int] = None,
) -> str:
    """
    Annotate screenshot with element bounding boxes and indices.

    Useful for debugging and visualization.

    Args:
        screenshot_b64: Base64-encoded screenshot.
        elements: List of UI elements with bounds.
        highlight_index: Optional element index to highlight.

    Returns:
        Base64-encoded annotated image.
    """
    image_bytes = base64.b64decode(screenshot_b64)
    image = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(image)

    # Try to use a readable font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for element in elements:
        bounds = element.get("bounds", {})
        index = element.get("index", -1)

        left = bounds.get("left", 0)
        top = bounds.get("top", 0)
        right = bounds.get("right", 0)
        bottom = bounds.get("bottom", 0)

        # Skip elements with no bounds
        if right <= left or bottom <= top:
            continue

        # Determine color
        if highlight_index is not None and index == highlight_index:
            color = (255, 0, 0)  # Red for highlighted
            width = 3
        elif element.get("clickable"):
            color = (0, 255, 0)  # Green for clickable
            width = 2
        else:
            color = (100, 100, 255)  # Blue for others
            width = 1

        # Draw bounding box
        draw.rectangle(
            [(left, top), (right, bottom)],
            outline=color,
            width=width,
        )

        # Draw index label
        label = str(index)
        label_bbox = draw.textbbox((left, top - 16), label, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]

        # Draw label background
        draw.rectangle(
            [(left, top - label_height - 4), (left + label_width + 4, top)],
            fill=color,
        )
        draw.text((left + 2, top - label_height - 2), label, fill=(255, 255, 255), font=font)

    # Save annotated image
    output = io.BytesIO()
    image.save(output, format="PNG")

    logger.debug("Screenshot annotated", element_count=len(elements))

    return base64.b64encode(output.getvalue()).decode("utf-8")


def get_image_dimensions(screenshot_b64: str) -> Tuple[int, int]:
    """
    Get dimensions of a base64-encoded image.

    Args:
        screenshot_b64: Base64-encoded image.

    Returns:
        Tuple of (width, height).
    """
    image_bytes = base64.b64decode(screenshot_b64)
    image = Image.open(io.BytesIO(image_bytes))
    return image.size


def crop_element(
    screenshot_b64: str,
    bounds: dict,
    padding: int = 10,
) -> str:
    """
    Crop screenshot to show a specific element.

    Args:
        screenshot_b64: Base64-encoded screenshot.
        bounds: Element bounds dict with left, top, right, bottom.
        padding: Pixels of padding around element.

    Returns:
        Base64-encoded cropped image.
    """
    image_bytes = base64.b64decode(screenshot_b64)
    image = Image.open(io.BytesIO(image_bytes))

    left = max(0, bounds.get("left", 0) - padding)
    top = max(0, bounds.get("top", 0) - padding)
    right = min(image.width, bounds.get("right", image.width) + padding)
    bottom = min(image.height, bounds.get("bottom", image.height) + padding)

    cropped = image.crop((left, top, right, bottom))

    output = io.BytesIO()
    cropped.save(output, format="PNG")

    return base64.b64encode(output.getvalue()).decode("utf-8")
