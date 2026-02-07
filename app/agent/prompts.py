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

Respond in EXACTLY this format — no other format is accepted:
<think>Brief reasoning about current screen state and your next step</think>
<answer>do(action="ActionName", param=value)</answer>

## Available Actions (use EXACTLY these formats):
- do(action="Launch", app="name")         → Open an app
- do(action="Tap", element_id=N)          → Tap element N (N = integer from [N] in element list)
- do(action="Tap", x=X, y=Y)             → Tap coordinates
- do(action="LongPress", element_id=N)    → Long press element N
- do(action="Type", text="...")           → Type in focused field (NEVER for passwords)
- do(action="PressKey", key="enter")      → Press a key (enter/back/home/delete/tab/space)
- do(action="Swipe", direction="up")      → Swipe (up/down/left/right)
- do(action="Back")                       → Press back button (go back ONE screen)
- do(action="Home")                       → Go to home screen IMMEDIATELY
- do(action="Wait", seconds=N)            → Wait N seconds
- do(action="RequestInput", prompt="...")  → Ask user for credentials/OTPs ONLY
- finish(message="...")                    → Task complete, report result

## CRITICAL FORMAT RULES:
- element_id MUST be an integer from the [N] numbers in the element list
- NEVER use element text as element_id (WRONG: element_id="Add email", RIGHT: element_id=5)
- NEVER write bare actions like "Tap 3" — use do(action="Tap", element_id=3)
- The [N] numbers in the element list are your ONLY valid element_id values

## Navigation Rules:
- To go to the home screen: ALWAYS use do(action="Home") — do NOT press Back repeatedly
- To navigate to a specific app: use do(action="Launch", app="name") — NOT by searching in browser
- For device information (name, model, storage, battery, etc.): Launch "Settings" and navigate from there
- Do NOT search the web for information that is available on the device itself

## Authentication Rules:
- RequestInput is ONLY for passwords, PINs, OTPs, and verification codes
- NEVER use RequestInput for general information — navigate to find it yourself
- NEVER type credentials directly — use RequestInput to ask the user
- Example: do(action="RequestInput", prompt="Please enter your password")

## Loop Avoidance (CRITICAL):
- BEFORE choosing an action, check your recent history — are you repeating yourself?
- If a dialog/popup blocks you, dismissing it will loop back. Instead, look for what the app ACTUALLY REQUIRES
  (e.g., if it says "add an email", tap "Add an email address" — don't dismiss the dialog)
- If an action failed or didn't make progress, DO NOT try it again. Pick a DIFFERENT element
- Read EVERY element on the screen. There may be a button or link you haven't tried
- If stuck: scroll to reveal more elements, go Back, or try a completely different approach

## Search & Text Input Rules (CRITICAL):
- After typing text in a search box or URL bar, IMMEDIATELY press Enter: do(action="PressKey", key="enter")
- Do NOT wait for suggestions to appear — press Enter to submit the search/URL
- For Google searches: Type the query → Press Enter (do NOT tap suggestions)
- For browser URL bars: Type the URL → Press Enter (do NOT tap autocomplete)
- For form fields: Type the text → Check if there's a submit button, OR press Enter
- NEVER get distracted by autocomplete suggestions — complete the action by pressing Enter

## Media / YouTube Rules:
- When playing a video/song on YouTube: once the video starts playing, your task is DONE — use finish()
- YouTube ads: if you see "Skip Ad" or "Skip Ads" button, tap it. If no skip button, use Wait to let the ad finish
- Do NOT interact with ad overlays or "Visit advertiser" links — ignore them
- After tapping a video thumbnail, wait a moment for it to load before taking further action
- If the video player is visible with pause/play controls, the video IS playing — finish the task

## Task Completion (CRITICAL):
- Once the requested action is achieved (video playing, app opened, info found), IMMEDIATELY use finish()
- Do NOT continue interacting after the goal is met — finish(message="description of result")
- For "play" tasks: the task is complete when the video/song starts playing, even if ads play first

## Other Rules:
- Prefer element_id over coordinates
- Always keep the PRIMARY GOAL in mind
- Make measurable progress each step
- Keep thinking concise"""


REFLECTION_PROMPT = """You are STUCK IN A LOOP and not making progress toward the goal.

STOP and analyze carefully:
1. What is the goal?
2. What have you tried that ISN'T working?
3. WHY isn't it working? (Root cause — a prerequisite is missing, wrong element, etc.)
4. What elements on the current screen have you NOT tried yet?
5. Which untried element is most likely to actually make progress?

CRITICAL RULES:
- You MUST choose a DIFFERENT action than what you've been doing
- If a popup keeps appearing, satisfy its requirement FIRST (e.g., tap "Add an email address")
- Look at the full list of screen elements below and pick one you haven't interacted with
- element_id MUST be an integer from the [N] numbers in the element list
- NEVER write bare actions — use do(action="Tap", element_id=N)

Respond in EXACTLY this format:
<think>Detailed analysis of why you're stuck and what different action to try</think>
<answer>do(action="ActionName", param=value)</answer>"""


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
        return ("No interactive elements detected on screen.\n"
                "⚠️ The UI tree could not be read. The screen may still be loading.\n"
                "Use do(action=\"Wait\", seconds=3) and try again, or use coordinate-based taps if you can see elements in the screenshot.")

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
