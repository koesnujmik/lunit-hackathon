from __future__ import annotations

from typing import Any

from .config import Settings
from .harness import L2Harness


def generate_answer(settings: Settings, conversation: list[dict[str, Any]]) -> str:
    return L2Harness(settings).generate(conversation)
