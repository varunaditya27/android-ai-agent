"""
UI Parser
=========

Parse and format Android UI accessibility tree into actionable elements.

Extracts meaningful UI elements from the accessibility tree XML,
filters out non-interactive/invisible elements, and formats them
for LLM consumption.

Usage:
    from app.perception import UIParser

    parser = UIParser()
    elements = parser.parse_hierarchy(ui_hierarchy)
    formatted = parser.format_for_llm(elements)
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
from xml.etree import ElementTree

from app.utils.logger import get_logger

logger = get_logger(__name__)


class ScreenType(Enum):
    """Types of screens detected."""

    HOME = auto()
    APP = auto()
    LOGIN = auto()
    SEARCH = auto()
    LIST = auto()
    DETAIL = auto()
    DIALOG = auto()
    KEYBOARD = auto()
    LOADING = auto()
    ERROR = auto()
    UNKNOWN = auto()


@dataclass
class UIElement:
    """
    Represents a UI element on the screen.

    Attributes:
        index: Unique index for LLM reference.
        element_class: Android class name.
        text: Visible text content.
        content_desc: Accessibility description.
        resource_id: Android resource ID.
        bounds: Bounding box coordinates.
        clickable: Whether element is clickable.
        enabled: Whether element is enabled.
        focusable: Whether element can receive focus.
        focused: Whether element is currently focused.
        scrollable: Whether element is scrollable.
        long_clickable: Whether long press is supported.
        checkable: Whether element is a checkbox/switch.
        checked: Whether checkbox is checked.
        editable: Whether element accepts text input.
    """

    index: int
    element_class: str = ""
    text: str = ""
    content_desc: str = ""
    resource_id: str = ""
    bounds: dict = field(default_factory=dict)
    clickable: bool = False
    enabled: bool = True
    focusable: bool = False
    focused: bool = False
    scrollable: bool = False
    long_clickable: bool = False
    checkable: bool = False
    checked: bool = False
    editable: bool = False

    @property
    def center_x(self) -> int:
        """Get center X coordinate."""
        return self.bounds.get("center_x", 0)

    @property
    def center_y(self) -> int:
        """Get center Y coordinate."""
        return self.bounds.get("center_y", 0)

    @property
    def display_text(self) -> str:
        """Get the best text to display for this element."""
        return self.text or self.content_desc or self._simplify_resource_id()

    def _simplify_resource_id(self) -> str:
        """Extract readable name from resource ID."""
        if not self.resource_id:
            return ""
        # Extract last part: com.app:id/button_name -> button_name
        parts = self.resource_id.split("/")
        if len(parts) > 1:
            return parts[-1].replace("_", " ").title()
        return ""

    @property
    def is_interactive(self) -> bool:
        """Check if element can be interacted with."""
        return (
            self.clickable
            or self.focusable
            or self.scrollable
            or self.checkable
            or self.editable
        ) and self.enabled

    @property
    def element_type(self) -> str:
        """Get simplified element type from class name."""
        class_lower = self.element_class.lower()

        if "button" in class_lower:
            return "Button"
        elif "edittext" in class_lower or "edit" in class_lower:
            return "Input"
        elif "textview" in class_lower or "text" in class_lower:
            return "Text"
        elif "imageview" in class_lower or "image" in class_lower:
            return "Image"
        elif "checkbox" in class_lower or "check" in class_lower:
            return "Checkbox"
        elif "switch" in class_lower or "toggle" in class_lower:
            return "Switch"
        elif "radiobutton" in class_lower:
            return "Radio"
        elif "recyclerview" in class_lower or "listview" in class_lower:
            return "List"
        elif "scrollview" in class_lower:
            return "ScrollView"
        elif "webview" in class_lower:
            return "WebView"
        elif "spinner" in class_lower:
            return "Dropdown"
        elif "seekbar" in class_lower or "slider" in class_lower:
            return "Slider"
        else:
            return "View"


class UIParser:
    """
    Parser for Android UI accessibility tree.

    Extracts meaningful UI elements from raw hierarchy data,
    filters based on visibility and interactivity, and provides
    LLM-friendly formatting.
    """

    def __init__(
        self,
        include_non_interactive: bool = False,
        max_elements: int = 50,
        min_element_size: int = 10,
    ) -> None:
        """
        Initialize UI parser.

        Args:
            include_non_interactive: Include non-interactive elements.
            max_elements: Maximum elements to return.
            min_element_size: Minimum element size in pixels.
        """
        self.include_non_interactive = include_non_interactive
        self.max_elements = max_elements
        self.min_element_size = min_element_size

    def parse_hierarchy(self, hierarchy: dict[str, Any]) -> list[UIElement]:
        """
        Parse UI hierarchy into list of UIElements.

        Args:
            hierarchy: Dictionary with 'xml' or 'elements' keys.

        Returns:
            List of parsed UIElement objects.
        """
        # If elements are already parsed, convert them
        if "elements" in hierarchy and hierarchy["elements"]:
            return self._convert_elements(hierarchy["elements"])

        # Parse from XML
        if "xml" in hierarchy and hierarchy["xml"]:
            return self._parse_xml(hierarchy["xml"])

        logger.warning("No UI data to parse")
        return []

    def _convert_elements(self, elements: list[dict]) -> list[UIElement]:
        """Convert raw element dictionaries to UIElement objects."""
        ui_elements = []

        for i, elem in enumerate(elements):
            if len(ui_elements) >= self.max_elements:
                break

            bounds = elem.get("bounds", {})

            # Skip elements that are too small
            width = bounds.get("width", 0)
            height = bounds.get("height", 0)
            if width < self.min_element_size or height < self.min_element_size:
                continue

            # Check if element is editable (EditText)
            element_class = elem.get("class", "")
            is_editable = "edit" in element_class.lower()

            ui_elem = UIElement(
                index=len(ui_elements),
                element_class=element_class,
                text=elem.get("text", ""),
                content_desc=elem.get("content_desc", ""),
                resource_id=elem.get("resource_id", ""),
                bounds=bounds,
                clickable=elem.get("clickable", False),
                enabled=elem.get("enabled", True),
                focusable=elem.get("focusable", False),
                focused=elem.get("focused", False),
                scrollable=elem.get("scrollable", False),
                long_clickable=elem.get("long_clickable", False),
                checkable=elem.get("checkable", False),
                checked=elem.get("checked", False),
                editable=is_editable,
            )

            # Filter based on criteria
            if self._should_include(ui_elem):
                ui_elements.append(ui_elem)

        logger.debug(
            "Parsed elements from hierarchy",
            total=len(elements),
            filtered=len(ui_elements),
        )

        return ui_elements

    def _parse_xml(self, xml_content: str) -> list[UIElement]:
        """Parse UI elements from XML accessibility tree."""
        ui_elements: list[UIElement] = []

        try:
            root = ElementTree.fromstring(xml_content)
            self._extract_from_xml_node(root, ui_elements)
        except ElementTree.ParseError as e:
            logger.error("Failed to parse UI XML", error=str(e))

        return ui_elements

    def _extract_from_xml_node(
        self,
        node: ElementTree.Element,
        elements: list[UIElement],
    ) -> None:
        """Recursively extract elements from XML node."""
        if len(elements) >= self.max_elements:
            return

        # Parse bounds
        bounds_str = node.attrib.get("bounds", "[0,0][0,0]")
        bounds = self._parse_bounds(bounds_str)

        # Skip elements that are too small
        if bounds["width"] < self.min_element_size or bounds["height"] < self.min_element_size:
            for child in node:
                self._extract_from_xml_node(child, elements)
            return

        element_class = node.attrib.get("class", "")
        is_editable = "edit" in element_class.lower()

        ui_elem = UIElement(
            index=len(elements),
            element_class=element_class,
            text=node.attrib.get("text", ""),
            content_desc=node.attrib.get("content-desc", ""),
            resource_id=node.attrib.get("resource-id", ""),
            bounds=bounds,
            clickable=node.attrib.get("clickable", "false") == "true",
            enabled=node.attrib.get("enabled", "true") == "true",
            focusable=node.attrib.get("focusable", "false") == "true",
            focused=node.attrib.get("focused", "false") == "true",
            scrollable=node.attrib.get("scrollable", "false") == "true",
            long_clickable=node.attrib.get("long-clickable", "false") == "true",
            checkable=node.attrib.get("checkable", "false") == "true",
            checked=node.attrib.get("checked", "false") == "true",
            editable=is_editable,
        )

        if self._should_include(ui_elem):
            elements.append(ui_elem)

        # Process children
        for child in node:
            self._extract_from_xml_node(child, elements)

    def _should_include(self, element: UIElement) -> bool:
        """Determine if element should be included."""
        # Always include interactive elements
        if element.is_interactive:
            return True

        # Include non-interactive elements with meaningful text
        if self.include_non_interactive and element.display_text:
            return True

        # Include elements with text that might be relevant
        if element.text or element.content_desc:
            return True

        return False

    def _parse_bounds(self, bounds_str: str) -> dict[str, int]:
        """Parse bounds string '[x1,y1][x2,y2]' to dict."""
        match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
        if match:
            x1, y1, x2, y2 = map(int, match.groups())
            return {
                "left": x1,
                "top": y1,
                "right": x2,
                "bottom": y2,
                "center_x": (x1 + x2) // 2,
                "center_y": (y1 + y2) // 2,
                "width": x2 - x1,
                "height": y2 - y1,
            }
        return {
            "left": 0, "top": 0, "right": 0, "bottom": 0,
            "center_x": 0, "center_y": 0, "width": 0, "height": 0,
        }

    def format_for_llm(self, elements: list[UIElement]) -> str:
        """
        Format elements into LLM-friendly text.

        Args:
            elements: List of UIElement objects.

        Returns:
            Formatted string describing UI elements.
        """
        if not elements:
            return "No interactive elements found on screen."

        lines = ["## Current UI Elements\n"]

        for elem in elements:
            # Build element description
            parts = [f"[{elem.index}]"]

            # Element type
            parts.append(elem.element_type)

            # Display text
            if elem.display_text:
                parts.append(f'"{elem.display_text}"')

            # Properties
            props = []
            if elem.clickable:
                props.append("clickable")
            if elem.editable:
                props.append("editable")
            if elem.scrollable:
                props.append("scrollable")
            if elem.checkable:
                if elem.checked:
                    props.append("checked")
                else:
                    props.append("unchecked")
            if elem.focused:
                props.append("focused")
            if not elem.enabled:
                props.append("disabled")

            if props:
                parts.append(f"[{', '.join(props)}]")

            lines.append(" ".join(parts))

        return "\n".join(lines)

    def detect_screen_type(self, elements: list[UIElement]) -> ScreenType:
        """
        Detect the type of screen based on elements.

        Args:
            elements: List of UIElement objects.

        Returns:
            Detected ScreenType.
        """
        # Collect all text content
        all_text = " ".join(
            (elem.text + " " + elem.content_desc).lower()
            for elem in elements
        )

        # Check for login indicators
        login_keywords = ["sign in", "log in", "login", "password", "email", "username", "forgot"]
        if any(kw in all_text for kw in login_keywords):
            return ScreenType.LOGIN

        # Check for search
        search_keywords = ["search", "find", "query"]
        if any(kw in all_text for kw in search_keywords):
            # Check if there's a search input
            for elem in elements:
                if elem.editable and "search" in (elem.text + elem.content_desc).lower():
                    return ScreenType.SEARCH

        # Check for dialogs/popups
        dialog_indicators = ["ok", "cancel", "dismiss", "allow", "deny"]
        resource_ids = " ".join(elem.resource_id.lower() for elem in elements)
        if "dialog" in resource_ids or "popup" in resource_ids:
            return ScreenType.DIALOG
        if sum(1 for kw in dialog_indicators if kw in all_text) >= 2:
            return ScreenType.DIALOG

        # Check for lists
        for elem in elements:
            if elem.scrollable and "recycler" in elem.element_class.lower():
                return ScreenType.LIST

        # Check for loading
        loading_keywords = ["loading", "please wait", "spinner"]
        if any(kw in all_text for kw in loading_keywords):
            return ScreenType.LOADING

        # Check for home screen
        if "launcher" in resource_ids:
            return ScreenType.HOME

        return ScreenType.APP

    def find_element_by_text(
        self,
        elements: list[UIElement],
        text: str,
        partial: bool = True,
    ) -> Optional[UIElement]:
        """
        Find an element by its text content.

        Args:
            elements: List of elements to search.
            text: Text to search for.
            partial: Allow partial matches.

        Returns:
            Matching UIElement or None.
        """
        text_lower = text.lower()

        for elem in elements:
            elem_text = (elem.text + " " + elem.content_desc).lower()

            if partial:
                if text_lower in elem_text:
                    return elem
            else:
                if text_lower == elem.text.lower() or text_lower == elem.content_desc.lower():
                    return elem

        return None

    def find_clickable_elements(self, elements: list[UIElement]) -> list[UIElement]:
        """Get all clickable elements."""
        return [elem for elem in elements if elem.clickable and elem.enabled]

    def find_editable_elements(self, elements: list[UIElement]) -> list[UIElement]:
        """Get all editable/input elements."""
        return [elem for elem in elements if elem.editable and elem.enabled]
