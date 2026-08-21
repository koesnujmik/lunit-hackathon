from __future__ import annotations

import json
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .answer import generate_answer
from .api import APIError
from .config import load_settings


DRIVER_MODEL_ID = "lunit-basic-driver"
MAX_REQUEST_BYTES = 2_000_000


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "LunitHackathonHTTP/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/v1/models":
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": DRIVER_MODEL_ID,
                            "object": "model",
                            "created": 0,
                            "owned_by": "lunit-hackathon",
                        }
                    ],
                },
            )
            return

        if path in {"/", "/health"}:
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found"}})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/v1/chat/completions":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found"}})
            return

        try:
            request_body = self._read_json_body()
            response_body = self._handle_chat_completions(request_body)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
            return
        except (RuntimeError, APIError) as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"message": str(exc)}})
            return

        self._send_json(HTTPStatus.OK, response_body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)

    def _read_json_body(self) -> dict[str, Any]:
        content_length_raw = self.headers.get("Content-Length")
        if not content_length_raw:
            raise ValueError("Missing Content-Length header.")

        try:
            content_length = int(content_length_raw)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc

        if content_length > MAX_REQUEST_BYTES:
            raise ValueError("Request body is too large.")

        raw_body = self.rfile.read(content_length).decode("utf-8")
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON.") from exc

        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object.")
        return body

    def _handle_chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list.")

        normalized_messages = [self._normalize_message(message) for message in messages]
        settings = load_settings()
        answer = generate_answer(settings, normalized_messages)
        response_model = body.get("model") if isinstance(body.get("model"), str) else DRIVER_MODEL_ID

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    @staticmethod
    def _normalize_message(message: Any) -> dict[str, str]:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object.")

        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str):
            raise ValueError(f"Unsupported message role: {role!r}.")

        if isinstance(content, str):
            normalized_content = content
        elif content is None:
            normalized_content = ""
        else:
            normalized_content = json.dumps(content, ensure_ascii=False)

        if role in {"system", "user", "assistant"}:
            normalized_role = role
        elif role == "developer":
            normalized_role = "system"
        else:
            normalized_role = "user"
            normalized_content = f"{role} message:\n{normalized_content}"

        return {"role": normalized_role, "content": normalized_content}

    def _send_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    host = "0.0.0.0"
    port = 8000
    server = ThreadingHTTPServer((host, port), ChatHandler)
    print(f"Serving on {host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
