"""
Tests for Perception Module
===========================

Comprehensive tests for:
- UIParser: parse_hierarchy (XML + dict), format_for_llm, detect_screen_type,
  find_element_by_text, find_clickable_elements, find_editable_elements
- UIElement: properties (center_x, center_y, display_text, is_interactive, element_type)
- AuthDetector: login, password-only, OTP, 2FA, OAuth, register, no-auth
- ElementDetector: hybrid detection from hierarchy
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.perception.ui_parser import UIParser, UIElement, ScreenType
from app.perception.auth_detector import AuthDetector, AuthType, AuthScreen
from app.perception.element_detector import ElementDetector
from tests.conftest import _make_bounds


# ===================================================================
# UIElement property tests
# ===================================================================


class TestUIElement:
    def test_center_coordinates(self):
        elem = UIElement(index=0, bounds=_make_bounds(0, 0, 100, 200))
        assert elem.center_x == 50
        assert elem.center_y == 100

    def test_display_text_prefers_text(self):
        elem = UIElement(index=0, text="Hello", content_desc="Alt", resource_id="com.app:id/name")
        assert elem.display_text == "Hello"

    def test_display_text_falls_back_to_content_desc(self):
        elem = UIElement(index=0, text="", content_desc="Description")
        assert elem.display_text == "Description"

    def test_display_text_falls_back_to_resource_id(self):
        elem = UIElement(index=0, text="", content_desc="", resource_id="com.app:id/submit_btn")
        assert "Submit" in elem.display_text or "submit" in elem.display_text.lower()

    def test_is_interactive(self):
        assert UIElement(index=0, clickable=True, enabled=True).is_interactive
        assert UIElement(index=0, editable=True, enabled=True).is_interactive
        assert UIElement(index=0, scrollable=True, enabled=True).is_interactive
        assert UIElement(index=0, checkable=True, enabled=True).is_interactive
        assert UIElement(index=0, focusable=True, enabled=True).is_interactive

    def test_not_interactive_when_disabled(self):
        elem = UIElement(index=0, clickable=True, enabled=False)
        assert not elem.is_interactive

    def test_element_type_button(self):
        assert UIElement(index=0, element_class="android.widget.Button").element_type == "Button"

    def test_element_type_input(self):
        assert UIElement(index=0, element_class="android.widget.EditText").element_type == "Input"

    def test_element_type_text(self):
        assert UIElement(index=0, element_class="android.widget.TextView").element_type == "Text"

    def test_element_type_image(self):
        assert UIElement(index=0, element_class="android.widget.ImageView").element_type == "Image"

    def test_element_type_checkbox(self):
        assert UIElement(index=0, element_class="android.widget.CheckBox").element_type == "Checkbox"

    def test_element_type_unknown(self):
        assert UIElement(index=0, element_class="com.custom.Widget").element_type == "View"

    def test_defaults(self):
        elem = UIElement(index=0)
        assert elem.clickable is False
        assert elem.editable is False
        assert elem.scrollable is False
        assert elem.focused is False
        assert elem.enabled is True
        assert elem.checked is False


# ===================================================================
# UIParser tests
# ===================================================================


class TestUIParser:
    def test_parse_from_xml(self, sample_ui_hierarchy_xml):
        parser = UIParser()
        elements = parser.parse_hierarchy({"xml": sample_ui_hierarchy_xml})
        assert len(elements) > 0
        assert all(isinstance(e, UIElement) for e in elements)

    def test_parse_xml_element_properties(self, sample_ui_hierarchy_xml):
        parser = UIParser()
        elements = parser.parse_hierarchy({"xml": sample_ui_hierarchy_xml})
        buttons = [e for e in elements if "Button" in e.element_class]
        assert len(buttons) >= 1
        assert buttons[0].clickable is True

    def test_parse_bounds_string(self):
        parser = UIParser()
        b = parser._parse_bounds("[100,200][300,400]")
        assert b["left"] == 100
        assert b["top"] == 200
        assert b["right"] == 300
        assert b["bottom"] == 400
        assert b["center_x"] == 200
        assert b["center_y"] == 300
        assert b["width"] == 200
        assert b["height"] == 200

    def test_parse_invalid_bounds(self):
        parser = UIParser()
        b = parser._parse_bounds("invalid")
        assert b["left"] == 0
        assert b["width"] == 0

    def test_parse_from_elements_dict(self):
        parser = UIParser()
        elements = parser.parse_hierarchy({
            "elements": [
                {
                    "class": "android.widget.Button",
                    "text": "OK",
                    "content_desc": "",
                    "resource_id": "",
                    "clickable": True,
                    "enabled": True,
                    "bounds": _make_bounds(0, 0, 200, 100),
                },
            ]
        })
        assert len(elements) == 1
        assert elements[0].text == "OK"
        assert elements[0].clickable is True

    def test_parse_empty_hierarchy(self):
        parser = UIParser()
        assert parser.parse_hierarchy({"xml": "<hierarchy></hierarchy>"}) == []
        assert parser.parse_hierarchy({}) == []
        assert parser.parse_hierarchy({"elements": []}) == []

    def test_parse_invalid_xml(self):
        parser = UIParser()
        assert parser.parse_hierarchy({"xml": "not valid xml"}) == []

    def test_filters_tiny_elements(self):
        parser = UIParser(min_element_size=20)
        elements = parser.parse_hierarchy({
            "elements": [
                {
                    "class": "android.widget.Button",
                    "text": "Tiny",
                    "clickable": True,
                    "enabled": True,
                    "bounds": _make_bounds(0, 0, 5, 5),  # too small
                },
                {
                    "class": "android.widget.Button",
                    "text": "Big",
                    "clickable": True,
                    "enabled": True,
                    "bounds": _make_bounds(0, 0, 200, 100),
                },
            ]
        })
        assert len(elements) == 1
        assert elements[0].text == "Big"

    def test_max_elements_cap(self):
        parser = UIParser(max_elements=3)
        elements = parser.parse_hierarchy({
            "elements": [
                {
                    "class": "android.widget.Button",
                    "text": f"Btn{i}",
                    "clickable": True,
                    "enabled": True,
                    "bounds": _make_bounds(0, i * 100, 200, i * 100 + 80),
                }
                for i in range(10)
            ]
        })
        assert len(elements) <= 3

    def test_editable_inferred_from_class(self):
        parser = UIParser()
        elements = parser.parse_hierarchy({
            "elements": [
                {
                    "class": "android.widget.EditText",
                    "text": "",
                    "clickable": True,
                    "enabled": True,
                    "bounds": _make_bounds(0, 0, 200, 100),
                },
            ]
        })
        assert elements[0].editable is True

    # --- Format for LLM ---

    def test_format_for_llm(self, sample_ui_hierarchy_xml):
        parser = UIParser()
        elements = parser.parse_hierarchy({"xml": sample_ui_hierarchy_xml})
        formatted = parser.format_for_llm(elements)
        assert isinstance(formatted, str)
        assert "[0]" in formatted

    def test_format_for_llm_empty(self):
        parser = UIParser()
        formatted = parser.format_for_llm([])
        assert "no" in formatted.lower() or "No" in formatted

    # --- Screen type detection ---

    def test_detect_login_screen(self):
        parser = UIParser()
        elements = [
            UIElement(index=0, text="Sign in", element_class="android.widget.TextView", bounds=_make_bounds(0, 0, 100, 50)),
            UIElement(index=1, text="Email", element_class="android.widget.EditText", editable=True, bounds=_make_bounds(0, 50, 100, 100)),
            UIElement(index=2, text="Password", element_class="android.widget.EditText", editable=True, bounds=_make_bounds(0, 100, 100, 150)),
        ]
        assert parser.detect_screen_type(elements) == ScreenType.LOGIN

    def test_detect_app_screen(self):
        parser = UIParser()
        elements = [
            UIElement(index=0, text="Welcome", element_class="android.widget.TextView", bounds=_make_bounds(0, 0, 100, 50)),
            UIElement(index=1, text="Start", element_class="android.widget.Button", clickable=True, bounds=_make_bounds(0, 50, 100, 100)),
        ]
        screen = parser.detect_screen_type(elements)
        assert screen in (ScreenType.APP, ScreenType.UNKNOWN)

    # --- Find helpers ---

    def test_find_element_by_text(self):
        parser = UIParser()
        elements = [
            UIElement(index=0, text="Submit", bounds=_make_bounds(0, 0, 100, 50)),
            UIElement(index=1, text="Cancel", bounds=_make_bounds(0, 50, 100, 100)),
        ]
        found = parser.find_element_by_text(elements, "Submit")
        assert found is not None
        assert found.index == 0

    def test_find_element_by_text_partial(self):
        parser = UIParser()
        elements = [
            UIElement(index=0, text="Submit Form", bounds=_make_bounds(0, 0, 100, 50)),
        ]
        found = parser.find_element_by_text(elements, "submit", partial=True)
        assert found is not None

    def test_find_element_not_found(self):
        parser = UIParser()
        assert parser.find_element_by_text([], "anything") is None

    def test_find_clickable_elements(self):
        parser = UIParser()
        elements = [
            UIElement(index=0, text="Label", bounds=_make_bounds(0, 0, 100, 50)),
            UIElement(index=1, text="Click Me", clickable=True, enabled=True, bounds=_make_bounds(0, 50, 100, 100)),
        ]
        clickable = parser.find_clickable_elements(elements)
        assert len(clickable) == 1
        assert clickable[0].index == 1

    def test_find_editable_elements(self):
        parser = UIParser()
        elements = [
            UIElement(index=0, text="Label", bounds=_make_bounds(0, 0, 100, 50)),
            UIElement(index=1, text="", editable=True, enabled=True, bounds=_make_bounds(0, 50, 100, 100)),
        ]
        editable = parser.find_editable_elements(elements)
        assert len(editable) == 1


# ===================================================================
# AuthDetector tests
# ===================================================================


class TestAuthDetector:
    def _elements(self, *specs) -> list[UIElement]:
        """Helper: specs are (text, editable, clickable, content_desc, resource_id)."""
        elems = []
        for i, s in enumerate(specs):
            text = s[0] if len(s) > 0 else ""
            editable = s[1] if len(s) > 1 else False
            clickable = s[2] if len(s) > 2 else False
            desc = s[3] if len(s) > 3 else ""
            rid = s[4] if len(s) > 4 else ""
            elems.append(UIElement(
                index=i,
                text=text,
                content_desc=desc,
                resource_id=rid,
                editable=editable,
                clickable=clickable,
                enabled=True,
                bounds=_make_bounds(0, i * 100, 500, i * 100 + 80),
            ))
        return elems

    def test_detect_login_screen(self):
        detector = AuthDetector()
        elements = self._elements(
            ("Sign in",),
            ("Email", True, False, "Enter your email"),
            ("Password", True, False, "Enter your password"),
            ("Login", False, True),
        )
        auth = detector.detect_auth(elements)
        assert auth is not None
        assert auth.auth_type == AuthType.LOGIN

    def test_detect_password_only(self):
        detector = AuthDetector()
        elements = self._elements(
            ("Enter your password",),
            ("", True, False, "password", "com.app:id/password_field"),
            ("Next", False, True),
        )
        auth = detector.detect_auth(elements)
        assert auth is not None
        assert auth.auth_type in (AuthType.PASSWORD_ONLY, AuthType.LOGIN)

    def test_detect_otp(self):
        detector = AuthDetector()
        elements = self._elements(
            ("Enter OTP",),
            ("", True),
            ("Verify", False, True),
        )
        auth = detector.detect_auth(elements)
        assert auth is not None
        assert auth.auth_type == AuthType.OTP

    def test_detect_2fa(self):
        detector = AuthDetector()
        elements = self._elements(
            ("Two-factor authentication",),
            ("", True),
            ("Submit", False, True),
        )
        auth = detector.detect_auth(elements)
        assert auth is not None
        assert auth.auth_type == AuthType.TWO_FACTOR

    def test_detect_register(self):
        detector = AuthDetector()
        elements = self._elements(
            ("Create account",),
            ("Name", True),
            ("Email", True),
            ("Password", True),
            ("Sign up", False, True),
        )
        auth = detector.detect_auth(elements)
        assert auth is not None
        assert auth.auth_type == AuthType.REGISTER

    def test_detect_oauth(self):
        detector = AuthDetector()
        elements = self._elements(
            ("Sign in with Google", False, True),
            ("Continue with Facebook", False, True),
            ("Sign in with Apple", False, True),
        )
        auth = detector.detect_auth(elements)
        assert auth is not None
        assert auth.auth_type == AuthType.OAUTH

    def test_no_auth_on_normal_screen(self):
        detector = AuthDetector()
        elements = self._elements(
            ("Welcome to App",),
            ("Get Started", False, True),
        )
        auth = detector.detect_auth(elements)
        assert auth is None

    def test_auth_screen_has_fields(self):
        detector = AuthDetector()
        elements = self._elements(
            ("Sign in",),
            ("", True, False, "Email"),
            ("", True, False, "Password"),
            ("Login", False, True),
        )
        auth = detector.detect_auth(elements)
        assert auth is not None
        assert len(auth.fields) >= 1

    def test_auth_screen_has_submit_button(self):
        detector = AuthDetector()
        elements = self._elements(
            ("Sign in",),
            ("", True, False, "Email"),
            ("Submit", False, True),
        )
        auth = detector.detect_auth(elements)
        assert auth is not None
        assert auth.submit_button is not None

    def test_confidence_score(self):
        detector = AuthDetector()
        elements = self._elements(
            ("Sign in",),
            ("Email", True, False, "email"),
            ("Password", True, False, "password"),
            ("Login", False, True),
        )
        auth = detector.detect_auth(elements)
        assert auth is not None
        assert auth.confidence > 0.5

    def test_credential_prompt_login(self):
        detector = AuthDetector()
        elements = self._elements(
            ("Sign in",),
            ("", True, False, "Email"),
            ("", True, False, "Password"),
            ("Login", False, True),
        )
        auth = detector.detect_auth(elements)
        prompt = detector.get_credential_prompt(auth)
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ===================================================================
# ElementDetector tests
# ===================================================================


class TestElementDetector:
    @pytest.mark.asyncio
    async def test_detect_from_hierarchy_dict(self, mock_llm_client):
        parser = UIParser()
        detector = ElementDetector(mock_llm_client, parser)

        elements = await detector.detect_elements(
            screenshot_b64="fake",
            ui_hierarchy={
                "elements": [
                    {
                        "class": "android.widget.Button",
                        "text": "OK",
                        "clickable": True,
                        "enabled": True,
                        "bounds": _make_bounds(0, 0, 200, 100),
                    }
                ]
            },
            use_vision_fallback=False,
        )
        assert len(elements) >= 1
        assert elements[0].text == "OK"

    @pytest.mark.asyncio
    async def test_elements_indexed_sequentially(self, mock_llm_client):
        parser = UIParser()
        detector = ElementDetector(mock_llm_client, parser)

        elements = await detector.detect_elements(
            screenshot_b64="fake",
            ui_hierarchy={
                "elements": [
                    {"class": "android.widget.Button", "text": f"B{i}", "clickable": True, "enabled": True, "bounds": _make_bounds(0, i*100, 200, i*100+80)}
                    for i in range(5)
                ]
            },
            use_vision_fallback=False,
        )
        indices = [e.index for e in elements]
        assert indices == list(range(len(elements)))


# ===================================================================
# ScreenType enum
# ===================================================================


class TestScreenType:
    def test_screen_types_exist(self):
        assert ScreenType.HOME is not None
        assert ScreenType.LOGIN is not None
        assert ScreenType.APP is not None
        assert ScreenType.UNKNOWN is not None
        assert ScreenType.DIALOG is not None
        assert ScreenType.LIST is not None
        assert ScreenType.LOADING is not None
