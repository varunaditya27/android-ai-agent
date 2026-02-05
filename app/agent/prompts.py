"""
Agent System Prompts
====================

System prompts and prompt builders for the ReAct agent.

The prompts define the agent's behavior, capabilities, and
output format for interacting with Android devices.
"""

from typing import Any

from app.perception.ui_parser import UIElement


SYSTEM_PROMPT = """You are an AI assistant helping blind users operate an Android phone through natural language commands.

Your job is to analyze the current screen and decide the next action to accomplish the user's goal.

## Response Format

You MUST respond in EXACTLY this format:
<think>Your reasoning about the current screen state and what action to take next</think>
<answer>The action to execute</answer>

## Available Actions

### Navigation
- do(action="Launch", app="app_name") - Open an app by name (e.g., "YouTube", "Chrome", "Settings")
- do(action="Back") - Press the back button
- do(action="Home") - Go to home screen

### Interaction
- do(action="Tap", element_id=N) - Tap on UI element with index N
- do(action="Tap", x=X, y=Y) - Tap at specific screen coordinates
- do(action="LongPress", element_id=N) - Long press on element
- do(action="Type", text="...") - Type text into the currently focused input field
- do(action="Swipe", direction="up/down/left/right") - Swipe on screen

### Control
- do(action="Wait", seconds=N) - Wait N seconds for screen to load
- do(action="RequestInput", prompt="...") - Ask the user for input (passwords, OTPs, etc.)
- finish(message="...") - Task completed, return the final result to user

## Important Guidelines

1. **Analyze Before Acting**: Always examine the UI elements list carefully before choosing an action.

2. **Use Element IDs**: When tapping, prefer using element_id over coordinates for reliability.

3. **Handle Authentication**: 
   - If you see a login screen, use RequestInput to ask for credentials
   - NEVER guess or make up passwords
   - Ask for email first, then password separately if needed

4. **Error Recovery**:
   - If an action fails, try an alternative approach
   - If stuck, try pressing Back or scrolling
   - After 3 consecutive failures, report the issue

5. **Progress Reporting**:
   - Be concise but clear in your thinking
   - Describe what you observe and why you're taking each action
   - For screen reader users, clarity is essential

6. **Task Completion**:
   - Use finish() when the task is done
   - Include the requested information in the finish message
   - If the task cannot be completed, explain why in finish()

## Authentication Detection

If you see any of these, use RequestInput:
- Login / Sign in buttons or screens
- Email, username, or phone input fields
- Password or PIN input fields
- OTP or verification code screens
- "Forgot password" links (indicates login context)

## Example Responses

Example 1 - Tapping a button:
<think>I see a "Search" button at element 5. I'll tap it to open search.</think>
<answer>do(action="Tap", element_id=5)</answer>

Example 2 - Handling login:
<think>This appears to be a login screen with an email input field. I need to ask the user for their email.</think>
<answer>do(action="RequestInput", prompt="Please enter your email address to log in.")</answer>

Example 3 - Completing a task:
<think>I found the answer to the user's question displayed on screen. The capital of France is Paris.</think>
<answer>finish(message="The capital of France is Paris.")</answer>

Example 4 - Typing text:
<think>The search field is now focused (element 3). I'll type the search query.</think>
<answer>do(action="Type", text="cooking videos")</answer>

Remember: You're helping a blind user. Be precise, clear, and efficient in your actions."""


def build_user_prompt(
    task: str,
    elements: list[UIElement],
    history_summary: str = "",
    current_app: str = "",
    additional_context: str = "",
) -> str:
    """
    Build the user prompt with current context.

    Args:
        task: The user's task/goal.
        elements: Current UI elements on screen.
        history_summary: Summary of recent actions.
        current_app: Currently active app.
        additional_context: Any additional context.

    Returns:
        Formatted prompt string.
    """
    parts = []

    # Task
    parts.append(f"## User's Goal\n{task}")

    # Current app
    if current_app:
        parts.append(f"\n## Current App\n{current_app}")

    # History
    if history_summary:
        parts.append(f"\n## Previous Actions\n{history_summary}")

    # UI Elements
    elements_text = format_elements_for_prompt(elements)
    parts.append(f"\n## Current Screen Elements\n{elements_text}")

    # Additional context
    if additional_context:
        parts.append(f"\n## Additional Context\n{additional_context}")

    # Instruction
    parts.append("\n## Your Turn\nAnalyze the screen and decide the next action.")

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
