from __future__ import annotations

from typing import Any

from .api import OpenAICompatibleClient, first_message_content
from .config import Settings
from .prompts import BASIC_MEDICAL_SYSTEM_PROMPT


def build_l2_messages(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"role": "system", "content": BASIC_MEDICAL_SYSTEM_PROMPT}, *conversation]


def generate_answer(settings: Settings, conversation: list[dict[str, Any]]) -> str:
    client = OpenAICompatibleClient(
        base_url=settings.fm_api_url,
        api_key=settings.fm_api_key,
        timeout_sec=settings.timeout_sec,
    )
    response = client.chat_completions(
        model=settings.fm_model,
        messages=build_l2_messages(conversation),
        temperature=0.2,
    )
    return first_message_content(response).strip()
