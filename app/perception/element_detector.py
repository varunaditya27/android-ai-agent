"""
Element Detector
================

Hybrid element detection combining accessibility tree and vision.

When the accessibility tree doesn't provide enough information
(e.g., custom views, canvas-based UIs), fall back to vision-based
detection using the LLM.

Usage:
    from app.perception import ElementDetector

    detector = ElementDetector(llm_client)
    elements = await detector.detect_elements(screenshot, ui_hierarchy)
"""

from typing import Any, Optional

from app.llm.client import LLMClient, RateLimitError
from app.perception.ui_parser import UIElement, UIParser
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ElementDetector:
    """
    Hybrid element detector using accessibility tree + vision.

    Primary: Accessibility tree (fast, accurate)
    Fallback: Vision-based detection (for custom UIs)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        ui_parser: Optional[UIParser] = None,
        min_elements_for_vision_fallback: int = 3,
    ) -> None:
        """
        Initialize element detector.

        Args:
            llm_client: LLM client for vision-based detection.
            ui_parser: UI parser for accessibility tree.
            min_elements_for_vision_fallback: Minimum elements before using vision.
        """
        self.llm = llm_client
        self.parser = ui_parser or UIParser()
        self.min_elements = min_elements_for_vision_fallback

    async def detect_elements(
        self,
        screenshot_b64: str,
        ui_hierarchy: dict[str, Any],
        use_vision_fallback: bool = True,
    ) -> list[UIElement]:
        """
        Detect UI elements using hybrid approach.

        1. First, parse accessibility tree
        2. If few elements found and vision enabled, use LLM to detect more

        Args:
            screenshot_b64: Base64-encoded screenshot.
            ui_hierarchy: UI accessibility tree data.
            use_vision_fallback: Whether to use vision as fallback.

        Returns:
            List of detected UIElement objects.
        """
        # Parse accessibility tree
        elements = self.parser.parse_hierarchy(ui_hierarchy)

        logger.debug(
            "Accessibility tree parsed",
            element_count=len(elements),
            interactive_count=sum(1 for e in elements if e.is_interactive),
        )

        # Check if we need vision fallback
        interactive_count = sum(1 for e in elements if e.is_interactive)

        if (
            use_vision_fallback
            and self.llm
            and interactive_count < self.min_elements
        ):
            logger.info(
                "Using vision fallback for element detection",
                accessibility_elements=len(elements),
            )
            vision_elements = await self._detect_with_vision(screenshot_b64, elements)
            elements = self._merge_elements(elements, vision_elements)

        return elements

    async def _detect_with_vision(
        self,
        screenshot_b64: str,
        existing_elements: list[UIElement],
    ) -> list[UIElement]:
        """
        Detect elements using vision/LLM.

        Args:
            screenshot_b64: Base64-encoded screenshot.
            existing_elements: Elements already detected from accessibility tree.

        Returns:
            List of vision-detected UIElement objects.
        """
        if not self.llm:
            return []

        prompt = """Analyze this Android screenshot and identify interactive UI elements.

For each element you find, provide:
1. A description (what it appears to be)
2. Approximate bounding box as percentages [left, top, right, bottom] (0-100)
3. Whether it appears clickable/interactive
4. Any visible text

Output as a JSON array:
```json
[
    {
        "description": "Search button with magnifying glass icon",
        "bounds": [85, 2, 98, 8],
        "clickable": true,
        "text": ""
    },
    {
        "description": "Email input field",
        "bounds": [10, 30, 90, 38],
        "clickable": true,
        "text": "Enter your email"
    }
]
```

Focus on:
- Buttons and clickable elements
- Input fields
- Navigation items
- Links and interactive text
- Icons that appear tappable

Be thorough but avoid duplicates. Return only the JSON array."""

        try:
            response = await self.llm.complete_with_vision(
                prompt=prompt,
                image_data=screenshot_b64,
                system_prompt="You are a UI analysis expert. Extract interactive elements from Android screenshots.",
            )

            vision_elements = self._parse_vision_response(response.content, existing_elements)
            logger.debug("Vision detection completed", element_count=len(vision_elements))
            return vision_elements

        except RateLimitError:
            # Re-raise rate limit error so parent can handle it
            raise
        except Exception as e:
            logger.warning("Vision detection failed", error=str(e))
            return []

    def _parse_vision_response(
        self,
        response: str,
        existing_elements: list[UIElement],
    ) -> list[UIElement]:
        """Parse vision detection response into UIElements."""
        import json
        import re

        # Extract JSON from response
        json_match = re.search(r"\[[\s\S]*\]", response)
        if not json_match:
            return []

        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to parse vision response JSON")
            return []

        # Start index after existing elements
        start_index = len(existing_elements)
        elements = []

        # Assume standard screen dimensions for percentage conversion
        # These will be adjusted based on actual device info
        screen_width = 1080
        screen_height = 2340

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            bounds_pct = item.get("bounds", [0, 0, 0, 0])
            if len(bounds_pct) != 4:
                continue

            # Convert percentages to pixels
            left = int(bounds_pct[0] * screen_width / 100)
            top = int(bounds_pct[1] * screen_height / 100)
            right = int(bounds_pct[2] * screen_width / 100)
            bottom = int(bounds_pct[3] * screen_height / 100)

            element = UIElement(
                index=start_index + i,
                element_class="VisionDetected",
                text=item.get("text", ""),
                content_desc=item.get("description", ""),
                resource_id="",
                bounds={
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "center_x": (left + right) // 2,
                    "center_y": (top + bottom) // 2,
                    "width": right - left,
                    "height": bottom - top,
                },
                clickable=item.get("clickable", True),
                enabled=True,
            )

            elements.append(element)

        return elements

    def _merge_elements(
        self,
        tree_elements: list[UIElement],
        vision_elements: list[UIElement],
    ) -> list[UIElement]:
        """
        Merge accessibility tree and vision elements, removing duplicates.

        Uses overlap detection to avoid duplicating elements.
        """
        if not vision_elements:
            return tree_elements

        merged = list(tree_elements)

        for vision_elem in vision_elements:
            # Check if this element overlaps significantly with any existing element
            is_duplicate = False

            for existing in tree_elements:
                overlap = self._calculate_overlap(vision_elem.bounds, existing.bounds)
                if overlap > 0.5:  # 50% overlap threshold
                    is_duplicate = True
                    break

            if not is_duplicate:
                # Re-index to avoid conflicts
                vision_elem.index = len(merged)
                merged.append(vision_elem)

        logger.debug(
            "Elements merged",
            tree_count=len(tree_elements),
            vision_count=len(vision_elements),
            total=len(merged),
        )

        return merged

    def _calculate_overlap(self, bounds1: dict, bounds2: dict) -> float:
        """
        Calculate overlap ratio between two bounding boxes.

        Returns:
            Overlap ratio (0.0 to 1.0).
        """
        # Get coordinates
        x1_left = bounds1.get("left", 0)
        x1_right = bounds1.get("right", 0)
        y1_top = bounds1.get("top", 0)
        y1_bottom = bounds1.get("bottom", 0)

        x2_left = bounds2.get("left", 0)
        x2_right = bounds2.get("right", 0)
        y2_top = bounds2.get("top", 0)
        y2_bottom = bounds2.get("bottom", 0)

        # Calculate intersection
        inter_left = max(x1_left, x2_left)
        inter_right = min(x1_right, x2_right)
        inter_top = max(y1_top, y2_top)
        inter_bottom = min(y1_bottom, y2_bottom)

        if inter_right <= inter_left or inter_bottom <= inter_top:
            return 0.0

        inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)

        # Calculate union
        area1 = (x1_right - x1_left) * (y1_bottom - y1_top)
        area2 = (x2_right - x2_left) * (y2_bottom - y2_top)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area
