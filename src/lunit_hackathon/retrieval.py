from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .api import (
    APIError,
    OpenAICompatibleClient,
    assistant_message_for_history,
    first_choice_message,
    message_tool_calls,
)
from .config import Settings
from .mcp import MCPClient, MCPError, citable_items_from_result, tool_result_text
from .prompts import RETRIEVAL_SYSTEM_PROMPT
from .routing import source_families


FINALIZE_RETRIEVAL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "finalize_retrieval",
        "description": "Submit the final citation selection and end the retrieval phase.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["sufficient", "partial", "no_evidence"],
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cite_uid": {"type": "string"},
                            "relevance_score": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["cite_uid", "relevance_score"],
                        "additionalProperties": False,
                    },
                },
                "note": {"type": "string"},
            },
            "required": ["status", "items"],
            "additionalProperties": False,
        },
    },
}


class ChatClient(Protocol):
    def chat_completions(self, **kwargs: Any) -> dict[str, Any]: ...


class RetrievalToolClient(Protocol):
    def list_tools(self) -> list[Any]: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CitationSelection:
    cite_uid: str
    relevance_score: float


@dataclass
class RetrievalResult:
    status: str
    items: list[tuple[CitationSelection, dict[str, Any]]] = field(default_factory=list)
    note: str = ""
    mcp_tool_calls: int = 0

    def as_generation_content(self, max_chars: int) -> str:
        sections = [f"status: {self.status}"]
        if self.note:
            sections.append(f"note: {self.note}")

        remaining = max_chars - sum(len(section) for section in sections)
        for index, (selection, evidence) in enumerate(self.items, start=1):
            serialized = json.dumps(evidence, ensure_ascii=False, indent=2)
            header = f"[{index}]\ncite_uid: {selection.cite_uid}\nrelevance_score: {selection.relevance_score:.2f}\ncontent:"
            available = max(0, remaining - len(header) - 2)
            if available == 0:
                break
            if len(serialized) > available:
                serialized = serialized[:available] + "\n[evidence truncated]"
            section = f"{header}\n{serialized}"
            sections.append(section)
            remaining -= len(section)

        if not self.items:
            sections.append("No citable evidence was selected.")
        return "\n\n".join(sections)


class RetrievalEngine:
    def __init__(
        self,
        settings: Settings,
        *,
        chat_client: ChatClient | None = None,
        mcp_client: RetrievalToolClient | None = None,
    ) -> None:
        self.settings = settings
        self.chat_client = chat_client or OpenAICompatibleClient(
            settings.fm_api_url,
            settings.fm_api_key,
            settings.timeout_sec,
        )
        self.mcp_client = mcp_client or MCPClient(
            settings.mcp_url,
            settings.fm_api_key,
            settings.timeout_sec,
        )

    def retrieve(self, query: str) -> RetrievalResult:
        mcp_tools = _select_mcp_tools(query, self.mcp_client.list_tools())
        tool_by_name = {tool.name: tool for tool in mcp_tools}
        all_tools = [tool.as_openai_tool() for tool in mcp_tools]
        all_tools.append(FINALIZE_RETRIEVAL_TOOL)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Retrieve evidence for this self-contained query:\n{query}\n\n"
                    f"MCP tool-call budget: {self.settings.max_retrieval_tool_calls}. "
                    "End by calling finalize_retrieval."
                ),
            },
        ]
        observed_items: dict[str, dict[str, Any]] = {}
        mcp_call_count = 0
        max_rounds = self.settings.max_retrieval_tool_calls + 4

        for _ in range(max_rounds):
            budget_exhausted = mcp_call_count >= self.settings.max_retrieval_tool_calls
            response = self.chat_client.chat_completions(
                model=self.settings.fm_model,
                messages=_compacted_messages(messages),
                temperature=0,
                extra_body={
                    "tools": [FINALIZE_RETRIEVAL_TOOL] if budget_exhausted else all_tools,
                    "tool_choice": (
                        {
                            "type": "function",
                            "function": {"name": "finalize_retrieval"},
                        }
                        if budget_exhausted
                        else "auto"
                    ),
                },
            )
            message = first_choice_message(response)
            tool_calls = message_tool_calls(message)
            messages.append(assistant_message_for_history(message))

            if not tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": "Do not answer in prose. Call finalize_retrieval now.",
                    }
                )
                continue

            for tool_call in tool_calls:
                call_id, name, arguments = _parse_tool_call(tool_call)
                if name == "finalize_retrieval":
                    finalized = _finalize_result(arguments, observed_items, mcp_call_count)
                    _log_retrieval_event(
                        "retrieval_finalized",
                        status=finalized.status,
                        selected_items=len(finalized.items),
                        mcp_tool_calls=mcp_call_count,
                    )
                    return finalized

                if budget_exhausted or mcp_call_count >= self.settings.max_retrieval_tool_calls:
                    messages.append(
                        _tool_message(call_id, {"error": "MCP tool-call budget exhausted; finalize now."})
                    )
                    continue

                if name not in tool_by_name:
                    messages.append(_tool_message(call_id, {"error": f"Unknown MCP tool: {name}"}))
                    continue

                try:
                    result = self.mcp_client.call_tool(name, arguments)
                    mcp_call_count += 1
                    new_items = citable_items_from_result(result)
                    observed_items.update(new_items)
                    content = tool_result_text(result, self.settings.max_tool_result_chars)
                    _log_retrieval_event(
                        "retrieval_tool_complete",
                        tool=name,
                        call_index=mcp_call_count,
                        new_citable_items=len(new_items),
                    )
                except MCPError as exc:
                    mcp_call_count += 1
                    content = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    _log_retrieval_event(
                        "retrieval_tool_error",
                        tool=name,
                        call_index=mcp_call_count,
                        error_type=type(exc).__name__,
                    )
                messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

        fallback = _fallback_result(observed_items, mcp_call_count)
        _log_retrieval_event(
            "retrieval_fallback",
            status=fallback.status,
            selected_items=len(fallback.items),
            mcp_tool_calls=mcp_call_count,
        )
        return fallback


def _parse_tool_call(tool_call: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    call_id = tool_call.get("id")
    function = tool_call.get("function")
    if not isinstance(call_id, str) or not isinstance(function, dict):
        raise APIError(f"Malformed tool call: {tool_call!r}")
    name = function.get("name")
    raw_arguments = function.get("arguments", "{}")
    if not isinstance(name, str) or not isinstance(raw_arguments, str):
        raise APIError(f"Malformed function tool call: {tool_call!r}")
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise APIError(f"Tool {name} returned invalid JSON arguments: {raw_arguments!r}") from exc
    if not isinstance(arguments, dict):
        raise APIError(f"Tool {name} arguments must be an object.")
    return call_id, name, arguments


def _tool_message(call_id: str, content: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(content, ensure_ascii=False),
    }


def _finalize_result(
    arguments: dict[str, Any],
    observed_items: dict[str, dict[str, Any]],
    mcp_call_count: int,
) -> RetrievalResult:
    status = arguments.get("status")
    if status not in {"sufficient", "partial", "no_evidence"}:
        status = "partial" if observed_items else "no_evidence"
    note = arguments.get("note")
    if not isinstance(note, str):
        note = ""

    materialized: list[tuple[CitationSelection, dict[str, Any]]] = []
    raw_items = arguments.get("items")
    if isinstance(raw_items, list):
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            cite_uid = raw_item.get("cite_uid")
            score = raw_item.get("relevance_score")
            if not isinstance(cite_uid, str) or cite_uid not in observed_items:
                continue
            if not isinstance(score, (int, float)):
                score = 0.5
            selection = CitationSelection(cite_uid, min(1.0, max(0.0, float(score))))
            materialized.append((selection, observed_items[cite_uid]))

    if status == "sufficient" and not materialized:
        status = "no_evidence"
        note = note or "The retrieval model selected no materializable cite_uid."
    return RetrievalResult(status, materialized, note, mcp_call_count)


def _fallback_result(
    observed_items: dict[str, dict[str, Any]], mcp_call_count: int
) -> RetrievalResult:
    materialized = [
        (CitationSelection(cite_uid, 0.5), item)
        for cite_uid, item in list(observed_items.items())[:3]
    ]
    if materialized:
        return RetrievalResult(
            "partial",
            materialized,
            "Retrieval ended before finalize_retrieval; returning observed citable evidence.",
            mcp_call_count,
        )
    return RetrievalResult(
        "no_evidence",
        [],
        "Retrieval ended without citable evidence.",
        mcp_call_count,
    )


def _select_mcp_tools(query: str, tools: list[Any]) -> list[Any]:
    families = source_families(query)
    selected_names: set[str] = set()

    if "guideline" in families:
        selected_names.update(tool.name for tool in tools if tool.name.startswith("index_"))

    if "hira" in families:
        selected_names.update(
            tool.name
            for tool in tools
            if tool.name.startswith("index_")
            or tool.name.startswith("openapi_hira_")
            or tool.name == "hira_updates_search"
        )

    if "drug" in families:
        selected_names.update(
            tool.name
            for tool in tools
            if tool.name == "adr_retrieve_drug_info" or tool.name.startswith("openapi_mfds_")
        )

    if "law" in families:
        selected_names.update(tool.name for tool in tools if tool.name.startswith("openapi_law_"))

    if "code" in families:
        selected_names.update(
            tool.name
            for tool in tools
            if tool.name.startswith("kcd_") or tool.name == "openapi_hira_disease_check_code"
        )

    if "research" in families:
        selected_names.update(tool.name for tool in tools if tool.name.startswith("rag_"))

    if "faers" in families:
        selected_names.update(
            {"rag_get_all_data_sources", "rag_get_data_source_detail", "rag_sql_query"}
        )

    if not selected_names:
        return tools
    selected = [tool for tool in tools if tool.name in selected_names]
    return selected or tools


def _compacted_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_indexes = [index for index, message in enumerate(messages) if message.get("role") == "tool"]
    latest_tool_index = tool_indexes[-1] if tool_indexes else -1
    compacted: list[dict[str, Any]] = []

    for index, message in enumerate(messages):
        copied = dict(message)
        content = copied.get("content")
        if copied.get("role") == "tool" and isinstance(content, str):
            limit = 12_000 if index == latest_tool_index else 3_000
            if len(content) > limit:
                copied["content"] = content[:limit] + "\n[older tool result compacted]"
        compacted.append(copied)
    return compacted


def _log_retrieval_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)
