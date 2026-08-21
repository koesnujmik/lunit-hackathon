from __future__ import annotations

import json
import time
from typing import Any, Protocol

from .api import APIError, OpenAICompatibleClient, first_message_content
from .config import Settings


SYSTEM_PROMPT = """You are a careful medical assistant.
Answer the latest user message using the conversation context.

- Match the user's language and level of expertise.
- Answer the question first, then give only useful reasoning and next steps.
- Put emergency actions first when dangerous symptoms are present.
- Do not overdiagnose or prescribe personalized treatment without enough information.
- State uncertainty and important limitations clearly.
- Do not invent citations or claim that you searched external sources.
- Be concise but complete.
"""


class ChatClient(Protocol):
    def chat_completions(self, **kwargs: Any) -> dict[str, Any]: ...


def generate_answer(
    settings: Settings,
    conversation: list[dict[str, Any]],
    *,
    client: ChatClient | None = None,
) -> str:
    chat_client = client or OpenAICompatibleClient(
        settings.fm_api_url,
        settings.fm_api_key,
        settings.timeout_sec,
    )
    attempts = [
        _messages(conversation, limit=10, chars_per_message=8_000),
        _messages(conversation, limit=4, chars_per_message=2_500),
    ]
    last_error: APIError | None = None

    for attempt, messages in enumerate(attempts, start=1):
        started = time.monotonic()
        try:
            response = chat_client.chat_completions(
                model=settings.fm_model,
                messages=messages,
                temperature=0,
                max_tokens=settings.max_tokens,
                extra_body={},
            )
            answer = first_message_content(response).strip()
            if not answer:
                raise APIError(
                    "L2 returned an empty answer.",
                    kind="empty_response",
                    retryable=True,
                )
            _log("l2_complete", attempt=attempt, elapsed_ms=_elapsed_ms(started))
            return answer
        except APIError as exc:
            last_error = exc
            will_retry = exc.retryable and attempt < len(attempts)
            _log(
                "l2_retry" if will_retry else "l2_failed",
                attempt=attempt,
                elapsed_ms=_elapsed_ms(started),
                error_type=type(exc).__name__,
                error_kind=exc.kind,
                status_code=exc.status_code,
                retryable=exc.retryable,
            )
            if not will_retry:
                break

    raise last_error or APIError("L2 did not return an answer.")


def _messages(
    conversation: list[dict[str, Any]], *, limit: int, chars_per_message: int
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in conversation[-limit:]:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            role = "user"
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        normalized.append({"role": role, "content": content[:chars_per_message]})
    return [{"role": "system", "content": SYSTEM_PROMPT}, *normalized]


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1_000)


def _log(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)
