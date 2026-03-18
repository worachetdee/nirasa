"""Thai chat template for Nirasa."""

from __future__ import annotations

import re

DEFAULT_SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วย AI ที่เป็นประโยชน์ "
    "ตอบคำถามเป็นภาษาไทยอย่างถูกต้องและเป็นมิตร"
)

# Chat format tokens
SYSTEM_TAG = "<|system|>"
USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"


def apply_chat_template(
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
    system_prompt: str | None = None,
) -> str:
    """Apply chat template to a list of messages.

    Format:
        <|system|>
        {system message}
        <|user|>
        {user message}
        <|assistant|>
        {assistant message}

    Args:
        messages: List of message dicts with "role" and "content" keys.
        add_generation_prompt: Whether to add assistant tag at the end.
        system_prompt: Override default system prompt.

    Returns:
        Formatted chat string.
    """
    parts: list[str] = []
    has_system = False

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            parts.append(f"{SYSTEM_TAG}\n{content}")
            has_system = True
        elif role == "user":
            parts.append(f"{USER_TAG}\n{content}")
        elif role == "assistant":
            parts.append(f"{ASSISTANT_TAG}\n{content}")

    # Add default system prompt if none provided
    if not has_system:
        prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        parts.insert(0, f"{SYSTEM_TAG}\n{prompt}")

    if add_generation_prompt:
        parts.append(f"{ASSISTANT_TAG}\n")

    return "\n".join(parts)


def parse_chat_messages(formatted_text: str) -> list[dict[str, str]]:
    """Parse a formatted chat string back into messages.

    Args:
        formatted_text: Chat string produced by apply_chat_template.

    Returns:
        List of message dicts with "role" and "content" keys.
    """
    messages: list[dict[str, str]] = []

    # Split by role tags
    pattern = re.compile(
        r"(<\|system\|>|<\|user\|>|<\|assistant\|>)\n?"
    )

    parts = pattern.split(formatted_text)

    current_role = None
    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part == SYSTEM_TAG:
            current_role = "system"
        elif part == USER_TAG:
            current_role = "user"
        elif part == ASSISTANT_TAG:
            current_role = "assistant"
        elif current_role is not None:
            messages.append({"role": current_role, "content": part})
            current_role = None

    return messages
