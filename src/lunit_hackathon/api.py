from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class APIError(RuntimeError):
    """Raised when an OpenAI-compatible API request fails."""


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, timeout_sec: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def chat_completions(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if extra_body:
            body.update(extra_body)

        return self._post_json("/v1/chat/completions", body)

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(3):
            request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                    raw = response.read().decode("utf-8")
                    return json.loads(raw)
            except urllib.error.HTTPError as exc:
                raw_error = exc.read().decode("utf-8", errors="replace")
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < 2:
                    last_error = exc
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise APIError(f"POST {url} failed with HTTP {exc.code}: {raw_error}") from exc
            except TimeoutError as exc:
                raise APIError(f"POST {url} timed out after {self.timeout_sec:g}s.") from exc
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise APIError(f"POST {url} failed: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise APIError(f"POST {url} returned invalid JSON.") from exc

        raise APIError(f"POST {url} failed after retries: {last_error}")


def first_message_content(response: dict[str, Any]) -> str:
    message = first_choice_message(response)
    content = message.get("content")

    if not isinstance(content, str):
        raise APIError(f"Expected string message content, got {type(content).__name__}.")
    return content


def first_choice_message(response: dict[str, Any]) -> dict[str, Any]:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise APIError(f"Unexpected chat completion response shape: {response!r}") from exc

    if not isinstance(message, dict):
        raise APIError(f"Expected message object, got {type(message).__name__}.")
    return message
