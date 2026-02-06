"""
Agent System Prompts
====================

System prompts and prompt builders for the ReAct agent.

The prompts define the agent's behavior, capabilities, and
output format for interacting with Android devices.
"""

from typing import Any

from app.perception.ui_parser import UIElement


SYSTEM_PROMPT = """You help blind users operate an Android phone via natural language.

Analyze the screenshot and UI elements, then decide the next action.

Respond in EXACTLY this format:
<think>Brief reasoning</think>
<answer>action</answer>

Actions:
- do(action="Launch", app="name") - Open app
- do(action="Tap", element_id=N) - Tap element N
- do(action="Tap", x=X, y=Y) - Tap coordinates
- do(action="LongPress", element_id=N)
- do(action="Type", text="...") - Type in focused field (NEVER use for passwords/credentials)
- do(action="Swipe", direction="up/down/left/right")
- do(action="Back") / do(action="Home")
- do(action="Wait", seconds=N)
- do(action="RequestInput", prompt="...") - Ask user for credentials/OTPs
- finish(message="...") - Task done, return result

CRITICAL: Authentication Rules
- If you see password, PIN, OTP, verification code, or any auth field → MUST use RequestInput
- NEVER type credentials directly - passwords/OTPs must come from RequestInput
- If field contains "password", "pin", "code", "verify" → use RequestInput
- Examples: "Please enter your password", "Please enter the verification code"

Loop Avoidance:
- If same action fails 2+ times, try a COMPLETELY different approach
- Don't alternate between same 2 actions repeatedly
- Check your progress - are you getting closer to the goal?
- If stuck, use Back or Home to reset and try new path

Other Rules:
- Prefer element_id over coordinates
- Always keep the PRIMARY GOAL in mind
- Make measurable progress each step
- Keep thinking concise"""


def build_user_prompt(
    task: str,
    elements: list[UIElement],
    history_summary: str = "",
    current_app: str = "",
    additional_context: str = "",
    progress_status: str = "",
    current_step: int = 0,
    max_steps: int = 30,
) -> str:
    """
    Build the user prompt with current context.

    Args:
        task: The user's task/goal.
        elements: Current UI elements on screen.
        history_summary: Summary of recent actions.
        current_app: Currently active app.
        additional_context: Any additional context.
        progress_status: What has been accomplished so far.
        current_step: Current step number.
        max_steps: Maximum allowed steps.

    Returns:
        Formatted prompt string.
    """
    parts = []

    # Task (emphasized to prevent forgetting the goal)
    parts.append(f"## PRIMARY GOAL\n{task}")
    parts.append(f"\nStep {current_step}/{max_steps}")

    # Progress status (what's been accomplished)
    if progress_status:
        parts.append(f"\n## Progress Status\n{progress_status}")

    # Current app
    if current_app:
        parts.append(f"\n## Current App\n{current_app}")

    # History
    if history_summary:
        parts.append(f"\n## {history_summary}")

    # UI Elements (limit to interactive elements to save tokens)
    interactive = [e for e in elements if e.is_interactive or e.display_text]
    # Cap at 40 elements max to keep prompt bounded
    if len(interactive) > 40:
        interactive = interactive[:40]
    elements_text = format_elements_for_prompt(interactive)
    parts.append(f"\n## Screen Elements\n{elements_text}")

    # Additional context
    if additional_context:
        parts.append(f"\n## Additional Context\n{additional_context}")

    # Instruction (with goal reminder)
    parts.append(f"\n## Your Turn\nBased on the PRIMARY GOAL above and your progress so far, decide the next action. Focus on making progress toward the goal.")

    return "\n".join(parts)


def format_elements_for_prompt(elements: list[UIElement]) -> str:
    """
    Format UI elements for the LLM prompt.

    Args:
        elements: List of UI elements.

    Returns:
        Formatted element list string.
    """
    if not elements:
        return "No interactive elements detected on screen."

    lines = []

    for elem in elements:
        # Build element description
        parts = [f"[{elem.index}]"]

        # Element type
        parts.append(elem.element_type)

        # Display text (truncate if long)
        if elem.display_text:
            text = elem.display_text
            if len(text) > 50:
                text = text[:47] + "..."
            parts.append(f'"{text}"')

        # Key properties
        props = []
        if elem.clickable:
            props.append("clickable")
        if elem.editable:
            props.append("editable/input")
        if elem.scrollable:
            props.append("scrollable")
        if elem.checkable:
            state = "checked" if elem.checked else "unchecked"
            props.append(state)
        if elem.focused:
            props.append("focused")
        if not elem.enabled:
            props.append("disabled")

        if props:
            parts.append(f"[{', '.join(props)}]")

        lines.append(" ".join(parts))

    return "\n".join(lines)


def build_auth_prompt(auth_type: str, current_field: str) -> str:
    """
    Build a prompt for authentication input.

    Args:
        auth_type: Type of auth (login, otp, etc.).
        current_field: The field needing input.

    Returns:
        User-friendly prompt string.
    """
    prompts = {
        "email": "Please enter your email address.",
        "username": "Please enter your username.",
        "password": "Please enter your password.",
        "phone": "Please enter your phone number.",
        "otp": "Please enter the verification code sent to you.",
        "2fa": "Please enter your two-factor authentication code.",
        "pin": "Please enter your PIN.",
        "captcha": "Please solve the captcha on screen.",
    }

    return prompts.get(current_field, f"Please enter your {current_field}.")


def build_error_recovery_prompt(
    error: str,
    failed_action: str,
    retry_count: int,
) -> str:
    """
    Build a prompt for error recovery.

    Args:
        error: The error that occurred.
        failed_action: The action that failed.
        retry_count: How many times we've retried.

    Returns:
        Context for error recovery.
    """
    return f"""
## Error Occurred
The previous action failed: {failed_action}
Error: {error}
Retry attempt: {retry_count}

Please try an alternative approach. Consider:
1. Checking if the element is still visible
2. Scrolling to find the element
3. Using coordinates instead of element_id
4. Going back and trying a different path
"""
