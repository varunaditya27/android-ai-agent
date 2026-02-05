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


class ActionType(Enum):
    """Types of actions the agent can perform."""

    TAP = auto()
    LONG_PRESS = auto()
    SWIPE = auto()
    TYPE = auto()
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
