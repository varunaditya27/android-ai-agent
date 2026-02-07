"""
Response Parser
===============

Parse LLM responses in the <think>/<answer> format.
Extracts reasoning and actions from structured responses.

Response Format:
    <think>Your reasoning about what to do next</think>
    <answer>do(action="Tap", element_id=5)</answer>

Supported Actions:
    - do(action="Launch", app="app_name")
    - do(action="Tap", element_id=N) or do(action="Tap", x=X, y=Y)
    - do(action="LongPress", element_id=N)
    - do(action="Swipe", direction="up/down/left/right")
    - do(action="Type", text="...")
    - do(action="PressKey", key="enter/back/home/delete/tab/space")
    - do(action="Back")
    - do(action="Home")
    - do(action="Wait", seconds=N)
    - do(action="RequestInput", prompt="...")
    - finish(message="...")
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


def _sanitize_element_id(params: dict[str, Any]) -> dict[str, Any]:
    """Ensure element_id is a valid non-negative integer, or remove it.

    This prevents string values like 'Add' (from mis-parsed
    'TAP element=Add an email address') from reaching the action handler
    and crashing on int() conversion or sorted().
    """
    if "element_id" not in params:
        return params
    val = params["element_id"]
    # Already an int — keep it
    if isinstance(val, int):
        return params
    # Numeric string — coerce
    if isinstance(val, str) and val.strip().isdigit():
        params["element_id"] = int(val.strip())
        return params
    # Float that is whole — coerce
    if isinstance(val, float) and val == int(val) and val >= 0:
        params["element_id"] = int(val)
        return params
    # Anything else (e.g. 'Add', 'Google', None) — invalid, remove it
    logger.warning(
        "Removing invalid element_id (not an integer)",
        element_id=val,
        element_id_type=type(val).__name__,
    )
    del params["element_id"]
    return params


class ActionType(Enum):
    """Types of actions the agent can perform."""

    TAP = auto()
    LONG_PRESS = auto()
    SWIPE = auto()
    TYPE = auto()
    PRESS_KEY = auto()
    LAUNCH = auto()
    BACK = auto()
    HOME = auto()
    WAIT = auto()
    SCROLL = auto()
    FINISH = auto()
    REQUEST_INPUT = auto()
    UNKNOWN = auto()


@dataclass
class ParsedAction:
    """
    Parsed action from LLM response.

    Note: Named ParsedAction to distinguish from device.ActionResult 
    which represents device action success/failure.

    Attributes:
        action_type: The type of action to perform.
        params: Action-specific parameters.
        raw: The original action string.
    """

    action_type: ActionType
    params: dict[str, Any] = field(default_factory=dict)
    raw: str = ""

    @property
    def is_terminal(self) -> bool:
        """Check if this action ends the task."""
        return self.action_type == ActionType.FINISH

    @property
    def requires_input(self) -> bool:
        """Check if this action requires user input."""
        return self.action_type == ActionType.REQUEST_INPUT


# Backwards compatibility alias
ActionResult = ParsedAction


@dataclass
class ParsedResponse:
    """
    Complete parsed response from LLM.

    Attributes:
        thinking: The reasoning/thinking section.
        action: The parsed action to perform.
        raw: The original full response.
    """

    thinking: str
    action: ParsedAction
    raw: str = ""


def parse_response(response: str) -> ParsedResponse:
    """
    Parse a full LLM response with <think> and <answer> tags.

    Args:
        response: The raw LLM response string.

    Returns:
        ParsedResponse with extracted thinking and action.

    Example:
        >>> response = '''
        ... <think>I see a search button, I should tap it.</think>
        ... <answer>do(action="Tap", element_id=3)</answer>
        ... '''
        >>> parsed = parse_response(response)
        >>> print(parsed.thinking)
        "I see a search button, I should tap it."
        >>> print(parsed.action.action_type)
        ActionType.TAP
    """
    # Extract thinking section
    thinking = ""
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    if think_match:
        thinking = think_match.group(1).strip()

    # Extract answer/action section
    action_str = ""
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        action_str = answer_match.group(1).strip()
    else:
        # Fallback: try to find action patterns directly
        action_patterns = [
            r"(do\s*\(.*?\))",
            r"(finish\s*\(.*?\))",
        ]
        for pattern in action_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                action_str = match.group(1).strip()
                break

    # Parse the action
    action = parse_action(action_str)

    logger.debug(
        "Parsed LLM response",
        thinking_length=len(thinking),
        action_type=action.action_type.name,
        has_params=bool(action.params),
    )

    return ParsedResponse(
        thinking=thinking,
        action=action,
        raw=response,
    )


def parse_action(action_str: str) -> ActionResult:
    """
    Parse an action string into an ActionResult.

    Supports formats:
        - do(action="Tap", element_id=5)
        - do(action="Type", text="hello world")
        - finish(message="Task completed")

    Args:
        action_str: The action string to parse.

    Returns:
        ActionResult with type and parameters.
    """
    if not action_str:
        return ActionResult(action_type=ActionType.UNKNOWN, raw=action_str)

    action_str = action_str.strip()

    # Strip leading punctuation/whitespace that the LLM sometimes prepends
    # e.g. ": do(action=..." or "- do(action=..." or "> do(action=..."
    action_str = re.sub(r'^[:\-\*>•#]+\s*', '', action_str).strip()

    # Parse finish() action
    finish_match = re.match(
        r'finish\s*\(\s*message\s*=\s*["\'](.+?)["\']\s*\)',
        action_str,
        re.DOTALL | re.IGNORECASE,
    )
    if finish_match:
        return ActionResult(
            action_type=ActionType.FINISH,
            params={"message": finish_match.group(1)},
            raw=action_str,
        )

    # Parse do() action
    do_match = re.match(r"do\s*\((.*)\)", action_str, re.DOTALL | re.IGNORECASE)
    if not do_match:
        # === FALLBACK PARSERS for non-standard LLM output ===
        #
        # The LLM sometimes produces actions outside the do() format.
        # We try these in order from most-specific to least-specific.

        # Map used by all fallback parsers
        _action_type_map = {
            "tap": ActionType.TAP, "click": ActionType.TAP,
            "longpress": ActionType.LONG_PRESS, "long_press": ActionType.LONG_PRESS,
            "swipe": ActionType.SWIPE, "scroll": ActionType.SCROLL,
            "type": ActionType.TYPE, "input": ActionType.TYPE,
            "presskey": ActionType.PRESS_KEY, "press_key": ActionType.PRESS_KEY,
            "launch": ActionType.LAUNCH, "open": ActionType.LAUNCH,
            "back": ActionType.BACK, "home": ActionType.HOME,
            "wait": ActionType.WAIT,
            "requestinput": ActionType.REQUEST_INPUT, "request_input": ActionType.REQUEST_INPUT,
        }

        _action_names_re = (
            r"(?:tap|click|longpress|long_press|swipe|scroll|type|input"
            r"|presskey|press_key|launch|open|back|home|wait|requestinput|request_input)"
        )

        # --- Fallback 1: function-call style  e.g. tap(element_id=3), tap(3) ---
        func_match = re.match(
            _action_names_re + r"\s*\((.*)\)",
            action_str,
            re.DOTALL | re.IGNORECASE,
        )
        if func_match:
            action_name = action_str[:func_match.start(1) - 1].strip().rstrip("(")
            inner = func_match.group(1).strip()
            # Handle tap(3) — bare number inside parens
            if re.fullmatch(r"\d+", inner):
                params = {"element_id": int(inner)}
            else:
                synthetic = f'action="{action_name}", {inner}' if inner else f'action="{action_name}"'
                params = _parse_params(synthetic)
                params.pop("action", None)
            if "element" in params and "element_id" not in params:
                params["element_id"] = params.pop("element")
            params = _sanitize_element_id(params)
            action_type = _action_type_map.get(action_name.lower(), ActionType.UNKNOWN)
            logger.info("Parsed function-call action format", original=action_str, action_type=action_type.name)
            return ActionResult(action_type=action_type, params=params, raw=action_str)

        # --- Fallback 2: action + bare number  e.g. "Tap 3", "tap 5" ---
        bare_num_match = re.match(
            _action_names_re + r"[\s,]+(?:element[_\s]*(?:id)?[\s=:]*)?([\d]+)\s*$",
            action_str,
            re.IGNORECASE,
        )
        if bare_num_match:
            action_name = re.match(_action_names_re, action_str, re.IGNORECASE).group(0)
            element_id = int(bare_num_match.group(1))
            action_type = _action_type_map.get(action_name.lower(), ActionType.UNKNOWN)
            logger.info("Parsed action+number format", original=action_str, action_type=action_type.name, element_id=element_id)
            return ActionResult(action_type=action_type, params={"element_id": element_id}, raw=action_str)

        # --- Fallback 3: action + key=value pairs  e.g. "Tap, element_id=3", "Swipe direction=up" ---
        # Guard: only match if the right side looks like proper key=value pairs
        # (not natural language like 'element=Add an email address')
        bare_kv_match = re.match(
            _action_names_re + r"[\s,]+(\w+=.+)",
            action_str,
            re.IGNORECASE,
        )
        if bare_kv_match:
            action_name = re.match(_action_names_re, action_str, re.IGNORECASE).group(0)
            rest = bare_kv_match.group(1).strip()
            synthetic = f'action="{action_name}", {rest}'
            params = _parse_params(synthetic)
            params.pop("action", None)
            if "element" in params and "element_id" not in params:
                params["element_id"] = params.pop("element")
            params = _sanitize_element_id(params)
            action_type = _action_type_map.get(action_name.lower(), ActionType.UNKNOWN)
            # If it was an element-targeting action but element_id was rejected,
            # fall through to other parsers rather than returning an invalid action
            if action_type in (ActionType.TAP, ActionType.LONG_PRESS) and not params.get("element_id") and "x" not in params:
                logger.info("Fallback 3 produced element action without valid element_id, skipping", original=action_str)
            else:
                logger.info("Parsed bare key=value action format", original=action_str, action_type=action_type.name)
                return ActionResult(action_type=action_type, params=params, raw=action_str)

        # --- Fallback 4: bare number "3", "5" → assume TAP element_id ---
        bare_number_match = re.fullmatch(r"(\d+)", action_str.strip())
        if bare_number_match:
            element_id = int(bare_number_match.group(1))
            logger.info("Parsed bare number as TAP", original=action_str, element_id=element_id)
            return ActionResult(action_type=ActionType.TAP, params={"element_id": element_id}, raw=action_str)

        # --- Fallback 5: bare action word with no params  e.g. "Back", "Home" ---
        bare_word_match = re.fullmatch(_action_names_re, action_str.strip(), re.IGNORECASE)
        if bare_word_match:
            action_name = bare_word_match.group(0)
            action_type = _action_type_map.get(action_name.lower(), ActionType.UNKNOWN)
            logger.info("Parsed bare action word", original=action_str, action_type=action_type.name)
            return ActionResult(action_type=action_type, params={}, raw=action_str)

        # Nothing matched — truly unparseable
        logger.warning("Could not parse action string", action_str=action_str)
        return ActionResult(action_type=ActionType.UNKNOWN, raw=action_str)

    params_str = do_match.group(1)
    params = _parse_params(params_str)

    # Determine action type from the "action" parameter
    action_name = params.get("action", "").lower()

    action_type_map = {
        "tap": ActionType.TAP,
        "click": ActionType.TAP,
        "longpress": ActionType.LONG_PRESS,
        "long_press": ActionType.LONG_PRESS,
        "swipe": ActionType.SWIPE,
        "scroll": ActionType.SCROLL,
        "type": ActionType.TYPE,
        "input": ActionType.TYPE,
        "presskey": ActionType.PRESS_KEY,
        "press_key": ActionType.PRESS_KEY,
        "launch": ActionType.LAUNCH,
        "open": ActionType.LAUNCH,
        "back": ActionType.BACK,
        "home": ActionType.HOME,
        "wait": ActionType.WAIT,
        "requestinput": ActionType.REQUEST_INPUT,
        "request_input": ActionType.REQUEST_INPUT,
    }

    action_type = action_type_map.get(action_name, ActionType.UNKNOWN)

    # Remove 'action' from params since we've processed it
    params.pop("action", None)

    # Normalize common param name variants
    if "element" in params and "element_id" not in params:
        params["element_id"] = params.pop("element")

    # Validate element_id is a proper integer
    params = _sanitize_element_id(params)

    return ActionResult(
        action_type=action_type,
        params=params,
        raw=action_str,
    )


def _parse_params(params_str: str) -> dict[str, Any]:
    """
    Parse parameter string into a dictionary.

    Handles formats like:
        action="Tap", element_id=5, text="hello"

    Args:
        params_str: The parameter string to parse.

    Returns:
        Dictionary of parameter names to values.
    """
    params: dict[str, Any] = {}

    # Pattern to match key=value pairs
    # Handles: key="value", key='value', key=123, key=True
    pattern = r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\d+(?:\.\d+)?)|(\w+))'

    for match in re.finditer(pattern, params_str):
        key = match.group(1)

        # Determine which group matched the value
        if match.group(2) is not None:  # Double-quoted string
            value: Any = match.group(2)
        elif match.group(3) is not None:  # Single-quoted string
            value = match.group(3)
        elif match.group(4) is not None:  # Number
            num_str = match.group(4)
            value = float(num_str) if "." in num_str else int(num_str)
        elif match.group(5) is not None:  # Bare word (True, False, etc.)
            word = match.group(5)
            if word.lower() == "true":
                value = True
            elif word.lower() == "false":
                value = False
            elif word.lower() == "none":
                value = None
            else:
                value = word
        else:
            value = None

        params[key] = value

    return params


def format_action_for_log(action: ActionResult) -> str:
    """
    Format an action for logging/display.

    Args:
        action: The action to format.

    Returns:
        Human-readable action description.
    """
    if action.action_type == ActionType.TAP:
        if "element_id" in action.params:
            return f"Tap element #{action.params['element_id']}"
        elif "x" in action.params and "y" in action.params:
            return f"Tap at ({action.params['x']}, {action.params['y']})"
        return "Tap"

    elif action.action_type == ActionType.TYPE:
        text = action.params.get("text", "")
        # Mask if it looks like a password
        if len(text) > 0:
            display_text = text[:20] + "..." if len(text) > 20 else text
        else:
            display_text = "(empty)"
        return f"Type: '{display_text}'"

    elif action.action_type == ActionType.SWIPE:
        direction = action.params.get("direction", "unknown")
        return f"Swipe {direction}"

    elif action.action_type == ActionType.LAUNCH:
        app = action.params.get("app", "unknown")
        return f"Launch app: {app}"

    elif action.action_type == ActionType.FINISH:
        message = action.params.get("message", "")
        return f"Finish: {message[:50]}..." if len(message) > 50 else f"Finish: {message}"

    elif action.action_type == ActionType.REQUEST_INPUT:
        prompt = action.params.get("prompt", "input needed")
        return f"Request input: {prompt}"

    return f"{action.action_type.name}: {action.params}"
