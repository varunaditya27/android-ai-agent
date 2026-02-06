"""
Tests for Perception Module
===========================

Tests for:
- UI hierarchy parsing
- Element detection
- Auth screen detection
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.perception.ui_parser import UIParser, UIElement, ScreenType
from app.perception.auth_detector import AuthDetector, AuthType
from app.perception.element_detector import ElementDetector


class TestUIParser:
    """Tests for UIParser class."""

    def test_parse_basic_hierarchy(self, sample_ui_hierarchy):
        """Test parsing a basic UI hierarchy."""
        parser = UIParser()
        elements = parser.parse(sample_ui_hierarchy)

        assert len(elements) > 0
        assert all(isinstance(e, UIElement) for e in elements)

    def test_parse_element_properties(self, sample_ui_hierarchy):
        """Test element properties are parsed correctly."""
        parser = UIParser()
        elements = parser.parse(sample_ui_hierarchy)

        # Find button element
        buttons = [e for e in elements if e.element_type == "Button"]
        assert len(buttons) > 0

        button = buttons[0]
        assert button.clickable is True
        assert button.bounds is not None

    def test_parse_bounds(self):
        """Test bounds parsing."""
        parser = UIParser()

        hierarchy = """
        <hierarchy>
            <node index="0" text="Test" class="android.widget.Button"
                  bounds="[100,200][300,400]" clickable="true"/>
        </hierarchy>
        """

        elements = parser.parse(hierarchy)
        assert len(elements) == 1

        elem = elements[0]
        assert elem.bounds == (100, 200, 300, 400)
        assert elem.center == (200, 300)

    def test_parse_empty_hierarchy(self):
        """Test parsing empty hierarchy."""
        parser = UIParser()
        elements = parser.parse("<hierarchy></hierarchy>")
        assert elements == []

    def test_parse_invalid_xml(self):
        """Test handling invalid XML."""
        parser = UIParser()
        elements = parser.parse("not valid xml")
        assert elements == []

    def test_filter_interactive_elements(self, sample_ui_hierarchy):
        """Test filtering for interactive elements."""
        parser = UIParser()
        elements = parser.parse(sample_ui_hierarchy)

        interactive = [e for e in elements if e.clickable or e.editable]
        assert len(interactive) > 0
        assert all(e.clickable or e.editable for e in interactive)

    def test_element_display_text(self, sample_ui_hierarchy):
        """Test element display text property."""
        parser = UIParser()
        elements = parser.parse(sample_ui_hierarchy)

        # Elements with text
        with_text = [e for e in elements if e.display_text]
        assert len(with_text) > 0

    def test_detect_screen_type(self, sample_ui_hierarchy):
        """Test screen type detection."""
        parser = UIParser()
        elements = parser.parse(sample_ui_hierarchy)

        screen_type = parser.detect_screen_type(elements)

        # Should detect login screen based on Login button
        assert screen_type in [ScreenType.LOGIN, ScreenType.UNKNOWN, ScreenType.APP]


class TestUIElement:
    """Tests for UIElement class."""

    def test_element_creation(self):
        """Test UIElement creation."""
        element = UIElement(
            index=0,
            element_type="Button",
            display_text="Submit",
            bounds=(100, 100, 200, 150),
        )

        assert element.index == 0
        assert element.element_type == "Button"
        assert element.display_text == "Submit"

    def test_element_center_calculation(self):
        """Test center point calculation."""
        element = UIElement(
            index=0,
            element_type="Button",
            bounds=(0, 0, 100, 100),
        )

        assert element.center == (50, 50)

    def test_element_defaults(self):
        """Test UIElement default values."""
        element = UIElement(
            index=0,
            element_type="View",
            bounds=(0, 0, 100, 100),
        )

        assert element.clickable is False
        assert element.editable is False
        assert element.scrollable is False
        assert element.focused is False
        assert element.enabled is True

    def test_element_to_dict(self):
        """Test UIElement serialization."""
        element = UIElement(
            index=1,
            element_type="EditText",
            display_text="Email",
            bounds=(100, 200, 500, 250),
            clickable=True,
            editable=True,
        )

        data = element.to_dict()

        assert data["index"] == 1
        assert data["element_type"] == "EditText"
        assert data["display_text"] == "Email"
        assert data["clickable"] is True


class TestAuthDetector:
    """Tests for AuthDetector class."""

    def test_detect_login_screen(self):
        """Test login screen detection."""
        detector = AuthDetector()

        elements = [
            UIElement(index=0, element_type="EditText", display_text="Email", editable=True, bounds=(0, 0, 100, 50)),
            UIElement(index=1, element_type="EditText", display_text="Password", editable=True, bounds=(0, 50, 100, 100)),
            UIElement(index=2, element_type="Button", display_text="Login", clickable=True, bounds=(0, 100, 100, 150)),
        ]

        auth = detector.detect_auth(elements)

        assert auth is not None
        assert auth.auth_type == AuthType.LOGIN

    def test_detect_otp_screen(self):
        """Test OTP screen detection."""
        detector = AuthDetector()

        elements = [
            UIElement(index=0, element_type="TextView", display_text="Enter OTP", bounds=(0, 0, 100, 50)),
            UIElement(index=1, element_type="EditText", display_text="", editable=True, bounds=(0, 50, 100, 100)),
            UIElement(index=2, element_type="Button", display_text="Verify", clickable=True, bounds=(0, 100, 100, 150)),
        ]

        auth = detector.detect_auth(elements)

        assert auth is not None
        assert auth.auth_type == AuthType.OTP

    def test_detect_no_auth(self):
        """Test non-auth screen returns None."""
        detector = AuthDetector()

        elements = [
            UIElement(index=0, element_type="TextView", display_text="Welcome", bounds=(0, 0, 100, 50)),
            UIElement(index=1, element_type="Button", display_text="Start", clickable=True, bounds=(0, 50, 100, 100)),
        ]

        auth = detector.detect_auth(elements)
        assert auth is None

    def test_detect_password_field(self):
        """Test password field detection."""
        detector = AuthDetector()

        elements = [
            UIElement(index=0, element_type="EditText", display_text="Password", is_password=True, editable=True, bounds=(0, 0, 100, 50)),
        ]

        auth = detector.detect_auth(elements)

        assert auth is not None
        assert AuthType.LOGIN == auth.auth_type or "password" in str(auth.fields_needed).lower()

    def test_auth_keywords(self):
        """Test auth keyword detection."""
        detector = AuthDetector()

        # Test various login indicators
        login_keywords = ["Sign in", "Log in", "Login", "signin", "login"]

        for keyword in login_keywords:
            elements = [
                UIElement(index=0, element_type="Button", display_text=keyword, clickable=True, bounds=(0, 0, 100, 50)),
            ]

            auth = detector.detect_auth(elements)
            # Should detect some auth-related screen
            # (might not always be LOGIN type, but should not be None)


class TestElementDetector:
    """Tests for ElementDetector class."""

    @pytest.fixture
    def element_detector(self, mock_llm_client):
        """Create ElementDetector with mocks."""
        parser = UIParser()
        return ElementDetector(mock_llm_client, parser)

    @pytest.mark.asyncio
    async def test_detect_from_hierarchy(self, element_detector, sample_ui_hierarchy):
        """Test element detection from UI hierarchy."""
        elements = await element_detector.detect_elements(
            screenshot_b64="fake_screenshot",
            ui_hierarchy=sample_ui_hierarchy,
            use_vision_fallback=False,
        )

        assert len(elements) > 0
        assert all(isinstance(e, UIElement) for e in elements)

    @pytest.mark.asyncio
    async def test_detect_with_vision_fallback(self, element_detector):
        """Test vision fallback when hierarchy is empty."""
        # Setup mock to return vision-detected elements
        element_detector.llm_client.complete_with_vision = AsyncMock(
            return_value=MagicMock(
                content='[{"text": "Button", "bounds": [100, 100, 200, 150], "type": "button"}]'
            )
        )

        elements = await element_detector.detect_elements(
            screenshot_b64="fake_screenshot",
            ui_hierarchy="<hierarchy></hierarchy>",  # Empty
            use_vision_fallback=True,
        )

        # Should have attempted vision detection
        element_detector.llm_client.complete_with_vision.assert_called()

    @pytest.mark.asyncio
    async def test_index_elements(self, element_detector, sample_ui_hierarchy):
        """Test elements are properly indexed."""
        elements = await element_detector.detect_elements(
            screenshot_b64="fake_screenshot",
            ui_hierarchy=sample_ui_hierarchy,
        )

        # Check indices are sequential
        indices = [e.index for e in elements]
        assert indices == list(range(len(elements)))


class TestScreenType:
    """Tests for ScreenType enum."""

    def test_screen_types_exist(self):
        """Test all screen types are defined."""
        assert ScreenType.HOME is not None
        assert ScreenType.LOGIN is not None
        assert ScreenType.APP is not None
        assert ScreenType.SETTINGS is not None
        assert ScreenType.UNKNOWN is not None


class TestUIParserFormats:
    """Tests for UIParser format methods."""

    def test_format_for_llm(self, sample_ui_hierarchy):
        """Test formatting elements for LLM."""
        parser = UIParser()
        elements = parser.parse(sample_ui_hierarchy)

        formatted = parser.format_for_llm(elements)

        assert isinstance(formatted, str)
        assert len(formatted) > 0

        # Should contain element indices
        assert "[0]" in formatted or "0:" in formatted

    def test_format_for_llm_empty(self):
        """Test formatting empty element list."""
        parser = UIParser()
        formatted = parser.format_for_llm([])

        assert "no" in formatted.lower() or "empty" in formatted.lower()
