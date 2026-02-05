"""
Authentication Detector
=======================

Detect login/authentication screens and identify credential fields.

Supports detection of:
- Login/Sign-in screens
- Registration screens
- Password entry
- OTP/2FA verification
- OAuth/SSO flows

Usage:
    from app.perception import AuthDetector, AuthType

    detector = AuthDetector()
    auth_screen = detector.detect_auth(elements)
    if auth_screen:
        print(f"Detected: {auth_screen.auth_type}")
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from app.perception.ui_parser import UIElement
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AuthType(Enum):
    """Types of authentication screens."""

    LOGIN = auto()          # Standard login with email/password
    REGISTER = auto()       # Registration/signup
    PASSWORD_ONLY = auto()  # Password entry (email already entered)
    OTP = auto()            # OTP/verification code
    TWO_FACTOR = auto()     # 2FA screen
    OAUTH = auto()          # OAuth provider selection
    CAPTCHA = auto()        # Captcha challenge
    BIOMETRIC = auto()      # Fingerprint/face ID prompt
    NONE = auto()           # Not an auth screen


@dataclass
class AuthField:
    """A detected authentication input field."""

    element: UIElement
    field_type: str  # "email", "password", "otp", "username", etc.
    is_filled: bool = False
    placeholder: str = ""


@dataclass
class AuthScreen:
    """
    Detected authentication screen information.

    Attributes:
        auth_type: Type of auth screen detected.
        fields: Input fields found on the screen.
        submit_button: The submit/login button if found.
        alternative_actions: Other auth options (forgot password, signup, etc.)
        confidence: Detection confidence (0.0-1.0).
    """

    auth_type: AuthType
    fields: list[AuthField] = field(default_factory=list)
    submit_button: Optional[UIElement] = None
    alternative_actions: list[UIElement] = field(default_factory=list)
    confidence: float = 0.0


class AuthDetector:
    """
    Detector for authentication-related screens.

    Uses heuristics based on element text, resource IDs, and
    screen composition to identify auth flows.
    """

    # Keywords indicating different auth types
    LOGIN_KEYWORDS = [
        "sign in", "log in", "login", "welcome back",
        "enter your email", "enter your password",
    ]

    REGISTER_KEYWORDS = [
        "sign up", "register", "create account", "join",
        "get started", "new account",
    ]

    PASSWORD_KEYWORDS = [
        "password", "passcode", "pin", "secret",
    ]

    EMAIL_KEYWORDS = [
        "email", "e-mail", "mail", "username", "phone",
        "mobile number", "user id",
    ]

    OTP_KEYWORDS = [
        "otp", "verification code", "verify", "confirm code",
        "enter code", "6-digit", "4-digit", "sms code",
    ]

    TWO_FACTOR_KEYWORDS = [
        "two-factor", "2fa", "authenticator", "security code",
        "backup code",
    ]

    SUBMIT_KEYWORDS = [
        "sign in", "log in", "login", "submit", "continue",
        "next", "verify", "confirm",
    ]

    def detect_auth(self, elements: list[UIElement]) -> Optional[AuthScreen]:
        """
        Detect if current screen is an authentication screen.

        Args:
            elements: List of UI elements on screen.

        Returns:
            AuthScreen if auth detected, None otherwise.
        """
        # Collect all text content for analysis
        all_text = self._collect_text(elements).lower()

        # Check for different auth types in order of specificity
        if self._is_otp_screen(all_text, elements):
            return self._build_auth_screen(AuthType.OTP, elements)

        if self._is_2fa_screen(all_text, elements):
            return self._build_auth_screen(AuthType.TWO_FACTOR, elements)

        if self._is_oauth_screen(all_text, elements):
            return self._build_auth_screen(AuthType.OAUTH, elements)

        if self._is_register_screen(all_text, elements):
            return self._build_auth_screen(AuthType.REGISTER, elements)

        if self._is_login_screen(all_text, elements):
            return self._build_auth_screen(AuthType.LOGIN, elements)

        if self._is_password_only_screen(all_text, elements):
            return self._build_auth_screen(AuthType.PASSWORD_ONLY, elements)

        return None

    def _collect_text(self, elements: list[UIElement]) -> str:
        """Collect all text from elements."""
        texts = []
        for elem in elements:
            if elem.text:
                texts.append(elem.text)
            if elem.content_desc:
                texts.append(elem.content_desc)
            if elem.resource_id:
                # Extract readable part from resource ID
                parts = elem.resource_id.split("/")
                if len(parts) > 1:
                    texts.append(parts[-1].replace("_", " "))
        return " ".join(texts)

    def _is_login_screen(self, text: str, elements: list[UIElement]) -> bool:
        """Check if this is a login screen."""
        # Must have login keywords
        has_login_keywords = any(kw in text for kw in self.LOGIN_KEYWORDS)

        # Must have at least one editable field
        has_input = any(elem.editable for elem in elements)

        # Should not be registration
        is_register = any(kw in text for kw in self.REGISTER_KEYWORDS[:3])

        return has_login_keywords and has_input and not is_register

    def _is_register_screen(self, text: str, elements: list[UIElement]) -> bool:
        """Check if this is a registration screen."""
        has_register_keywords = any(kw in text for kw in self.REGISTER_KEYWORDS)
        has_input = any(elem.editable for elem in elements)
        return has_register_keywords and has_input

    def _is_password_only_screen(self, text: str, elements: list[UIElement]) -> bool:
        """Check if this is a password-only entry screen."""
        # Has password field but no email field
        has_password = any(kw in text for kw in self.PASSWORD_KEYWORDS)
        has_email = any(kw in text for kw in self.EMAIL_KEYWORDS)

        # Count editable fields - should be just one (password)
        editable_count = sum(1 for elem in elements if elem.editable)

        return has_password and not has_email and editable_count == 1

    def _is_otp_screen(self, text: str, elements: list[UIElement]) -> bool:
        """Check if this is an OTP verification screen."""
        has_otp_keywords = any(kw in text for kw in self.OTP_KEYWORDS)

        # OTP screens often have multiple small input fields or one field
        editable_elements = [elem for elem in elements if elem.editable]

        # Check for digit-only pattern (multiple single-digit inputs)
        if len(editable_elements) >= 4 and len(editable_elements) <= 8:
            # Could be individual digit inputs
            return has_otp_keywords

        return has_otp_keywords and len(editable_elements) > 0

    def _is_2fa_screen(self, text: str, elements: list[UIElement]) -> bool:
        """Check if this is a 2FA screen."""
        return any(kw in text for kw in self.TWO_FACTOR_KEYWORDS)

    def _is_oauth_screen(self, text: str, elements: list[UIElement]) -> bool:
        """Check if this is an OAuth provider selection screen."""
        oauth_providers = [
            "google", "facebook", "apple", "twitter", "github",
            "microsoft", "sign in with", "continue with",
        ]
        oauth_count = sum(1 for provider in oauth_providers if provider in text)
        return oauth_count >= 2

    def _build_auth_screen(
        self,
        auth_type: AuthType,
        elements: list[UIElement],
    ) -> AuthScreen:
        """Build AuthScreen object with detected fields."""
        fields = []
        submit_button = None
        alternatives = []

        for elem in elements:
            elem_text = (elem.text + " " + elem.content_desc + " " + elem.resource_id).lower()

            # Detect input fields
            if elem.editable:
                field_type = self._classify_input_field(elem_text, auth_type)
                fields.append(AuthField(
                    element=elem,
                    field_type=field_type,
                    is_filled=bool(elem.text and len(elem.text) > 0),
                    placeholder=elem.content_desc or elem.text,
                ))

            # Detect submit button
            elif elem.clickable:
                if any(kw in elem_text for kw in self.SUBMIT_KEYWORDS):
                    if submit_button is None:
                        submit_button = elem
                    else:
                        alternatives.append(elem)

                # Detect alternative actions
                elif any(kw in elem_text for kw in ["forgot", "help", "trouble", "skip"]):
                    alternatives.append(elem)

        confidence = self._calculate_confidence(auth_type, fields, submit_button)

        logger.info(
            "Auth screen detected",
            auth_type=auth_type.name,
            field_count=len(fields),
            has_submit=submit_button is not None,
            confidence=confidence,
        )

        return AuthScreen(
            auth_type=auth_type,
            fields=fields,
            submit_button=submit_button,
            alternative_actions=alternatives,
            confidence=confidence,
        )

    def _classify_input_field(self, text: str, auth_type: AuthType) -> str:
        """Classify the type of input field."""
        if any(kw in text for kw in self.PASSWORD_KEYWORDS):
            return "password"
        elif any(kw in text for kw in self.EMAIL_KEYWORDS):
            return "email"
        elif any(kw in text for kw in self.OTP_KEYWORDS):
            return "otp"
        elif auth_type == AuthType.OTP:
            return "otp"
        elif "name" in text:
            return "name"
        elif "phone" in text or "mobile" in text:
            return "phone"
        else:
            return "unknown"

    def _calculate_confidence(
        self,
        auth_type: AuthType,
        fields: list[AuthField],
        submit_button: Optional[UIElement],
    ) -> float:
        """Calculate confidence score for detection."""
        confidence = 0.5  # Base confidence

        # Having appropriate fields increases confidence
        if auth_type == AuthType.LOGIN:
            has_email = any(f.field_type == "email" for f in fields)
            has_password = any(f.field_type == "password" for f in fields)
            if has_email:
                confidence += 0.2
            if has_password:
                confidence += 0.2

        elif auth_type == AuthType.OTP:
            has_otp = any(f.field_type == "otp" for f in fields)
            if has_otp or len(fields) >= 4:
                confidence += 0.3

        elif auth_type == AuthType.PASSWORD_ONLY:
            if len(fields) == 1 and fields[0].field_type == "password":
                confidence += 0.3

        # Having a submit button increases confidence
        if submit_button:
            confidence += 0.1

        return min(confidence, 1.0)

    def get_credential_prompt(self, auth_screen: AuthScreen) -> str:
        """
        Generate a user-friendly prompt for credential input.

        Args:
            auth_screen: Detected auth screen info.

        Returns:
            Prompt string for the user.
        """
        if auth_screen.auth_type == AuthType.LOGIN:
            # Check which fields need input
            needs_email = any(
                f.field_type == "email" and not f.is_filled
                for f in auth_screen.fields
            )
            needs_password = any(
                f.field_type == "password" and not f.is_filled
                for f in auth_screen.fields
            )

            if needs_email and needs_password:
                return "Please enter your email address and password."
            elif needs_email:
                return "Please enter your email address."
            elif needs_password:
                return "Please enter your password."

        elif auth_screen.auth_type == AuthType.PASSWORD_ONLY:
            return "Please enter your password."

        elif auth_screen.auth_type == AuthType.OTP:
            return "Please enter the verification code sent to you."

        elif auth_screen.auth_type == AuthType.TWO_FACTOR:
            return "Please enter your two-factor authentication code."

        elif auth_screen.auth_type == AuthType.REGISTER:
            return "Please provide your registration details."

        return "Please provide the required credentials."
