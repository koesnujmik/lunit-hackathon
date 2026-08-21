from __future__ import annotations

import json
import time
from typing import Any, Protocol

from .api import (
    APIError,
    OpenAICompatibleClient,
    assistant_message_for_history,
    first_choice_message,
    message_tool_calls,
)
from .config import Settings
from .mcp import MCPError
from .prompts import DIRECT_GENERATION_SYSTEM_PROMPT, GENERATION_SYSTEM_PROMPT
from .retrieval import RetrievalEngine, RetrievalResult
from .routing import (
    conversation_user_text,
    explicitly_requests_retrieval,
    latest_user_text,
    source_families,
)


RETRIEVE_RELEVANT_CONTENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_relevant_content",
        "description": (
            "Retrieve authoritative content to ground the final answer. "
            "Pass one self-contained query with references from prior turns resolved."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A single self-contained retrieval query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


class ChatClient(Protocol):
    def chat_completions(self, **kwargs: Any) -> dict[str, Any]: ...


class EvidenceRetriever(Protocol):
    def retrieve(self, query: str) -> RetrievalResult: ...


class L2Harness:
    def __init__(
        self,
        settings: Settings,
        *,
        chat_client: ChatClient | None = None,
        retriever: EvidenceRetriever | None = None,
    ) -> None:
        self.settings = settings
        self.chat_client = chat_client or OpenAICompatibleClient(
            settings.fm_api_url,
            settings.fm_api_key,
            settings.timeout_sec,
        )
        self.retriever = retriever or RetrievalEngine(settings)

    def generate(self, conversation: list[dict[str, Any]]) -> str:
        started = time.monotonic()
        force_retrieval = explicitly_requests_retrieval(conversation)
        conversation_families = source_families(conversation_user_text(conversation))
        direct_fast_path = (
            self.settings.routing_mode == "hybrid"
            and not force_retrieval
            and not conversation_families
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    DIRECT_GENERATION_SYSTEM_PROMPT
                    if direct_fast_path
                    else GENERATION_SYSTEM_PROMPT
                ),
            },
            *conversation,
        ]
        retrieval_count = 0
        retrieval_status = "not_used"
        mcp_tool_calls = 0

        for _ in range(3):
            tools_enabled = self.settings.enable_retrieval and retrieval_count == 0
            extra_body: dict[str, Any] = {}
            if self.settings.enable_retrieval and not direct_fast_path:
                if tools_enabled and force_retrieval:
                    tool_choice: Any = {
                        "type": "function",
                        "function": {"name": "retrieve_relevant_content"},
                    }
                elif (
                    tools_enabled
                    and (self.settings.routing_mode == "model" or conversation_families)
                ):
                    tool_choice = "auto"
                else:
                    tool_choice = "none"
                extra_body = {
                    "tools": [RETRIEVE_RELEVANT_CONTENT_TOOL],
                    "tool_choice": tool_choice,
                }

            response = self.chat_client.chat_completions(
                model=self.settings.fm_model,
                messages=messages,
                temperature=0,
                extra_body=extra_body,
            )
            message = first_choice_message(response)
            tool_calls = message_tool_calls(message)

            if not tool_calls:
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise APIError("Generation L2 returned neither content nor a tool call.")
                _log_trace(retrieval_status, retrieval_count, mcp_tool_calls, started)
                return content.strip()

            messages.append(assistant_message_for_history(message))
            for tool_call in tool_calls:
                call_id, name, arguments = _parse_generation_tool_call(tool_call)
                if name != "retrieve_relevant_content":
                    messages.append(_tool_error(call_id, f"Unknown generation tool: {name}"))
                    continue
                if not tools_enabled:
                    messages.append(_tool_error(call_id, "Only one retrieval request is allowed."))
                    continue

                query = arguments.get("query")
                if not isinstance(query, str) or not query.strip():
                    messages.append(_tool_error(call_id, "query must be a non-empty string."))
                    continue

                retrieval_count += 1
                retrieval_query = query.strip()
                if force_retrieval and not source_families(retrieval_query):
                    retrieval_query = (
                        f"{retrieval_query}\n\nSource requirement from the user: "
                        f"{latest_user_text(conversation)}"
                    )

                if not force_retrieval and not source_families(retrieval_query):
                    retrieval = RetrievalResult(
                        "no_evidence",
                        note=(
                            "The query did not identify an authoritative source family. "
                            "Answer this general question from medical knowledge."
                        ),
                    )
                else:
                    try:
                        retrieval = self.retriever.retrieve(retrieval_query)
                    except (APIError, MCPError, RuntimeError) as exc:
                        retrieval = RetrievalResult(
                            "no_evidence",
                            note=f"Retrieval failed: {type(exc).__name__}",
                        )
                retrieval_status = retrieval.status
                mcp_tool_calls += retrieval.mcp_tool_calls
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": retrieval.as_generation_content(
                            self.settings.max_evidence_chars
                        ),
                    }
                )

        raise APIError("Generation L2 did not produce a final answer within the call budget.")


def _parse_generation_tool_call(
    tool_call: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    call_id = tool_call.get("id")
    function = tool_call.get("function")
    if not isinstance(call_id, str) or not isinstance(function, dict):
        raise APIError(f"Malformed generation tool call: {tool_call!r}")
    name = function.get("name")
    raw_arguments = function.get("arguments", "{}")
    if not isinstance(name, str) or not isinstance(raw_arguments, str):
        raise APIError(f"Malformed generation function call: {tool_call!r}")
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise APIError(f"Generation tool returned invalid JSON: {raw_arguments!r}") from exc
    if not isinstance(arguments, dict):
        raise APIError("Generation tool arguments must be an object.")
    return call_id, name, arguments


def _tool_error(call_id: str, message: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps({"status": "no_evidence", "error": message}),
    }


def _log_trace(
    retrieval_status: str,
    retrieval_count: int,
    mcp_tool_calls: int,
    started: float,
) -> None:
    print(
        json.dumps(
            {
                "event": "harness_complete",
                "retrieval_status": retrieval_status,
                "retrieval_calls": retrieval_count,
                "mcp_tool_calls": mcp_tool_calls,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            }
        ),
        flush=True,
    )
