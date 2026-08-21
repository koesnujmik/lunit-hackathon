import asyncio
import json
from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI

from .config import Settings
from .mcp_client import LunitMCPClient
from .models import CitationSelection, Evidence, RetrievalResult
from .prompts import GENERATION_SYSTEM_PROMPT, RETRIEVAL_SYSTEM_PROMPT
from .ranking import rank_documents


def _decision_tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


RETRIEVE_TOOL = _decision_tool(
    "retrieve_relevant_content",
    "Retrieve evidence for claims that require an external source. Pass one self-contained query.",
    {"query": {"type": "string", "minLength": 1}},
)


def _arguments(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"L2 returned invalid tool arguments: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise TypeError("L2 tool arguments must be a JSON object")
    return value


def _assistant_message(message: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        data["tool_calls"] = [call.model_dump(mode="json") for call in message.tool_calls]
    return data


def _response_language_instruction(text: str) -> str:
    if any("가" <= character <= "힣" for character in text):
        return "The required response language is Korean."
    if any(character.isascii() and character.isalpha() for character in text):
        return "The required response language is English. Respond in English only."
    return "Respond in the same language as the user's latest message."


def _conversation_context(messages: list[dict[str, str]], max_chars: int = 12_000) -> str:
    rendered = "\n\n".join(
        f"{message.get('role', 'user').upper()}: {message.get('content', '')}"
        for message in messages
    )
    return rendered[-max_chars:]


def _extract_evidence(tool_outputs: list[str], selection: CitationSelection) -> list[Evidence]:
    """Retained for compatibility with explicit citation selections."""
    evidence: list[Evidence] = []
    for selected in selection.items:
        matching = [text for text in tool_outputs if selected.cite_uid in text]
        if matching:
            evidence.append(
                Evidence(
                    cite_uid=selected.cite_uid,
                    relevance_score=selected.relevance_score,
                    content="\n".join(matching),
                )
            )
    return evidence


class L2Harness:
    def __init__(
        self,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
        mcp_factory: Callable[..., LunitMCPClient] = LunitMCPClient,
    ) -> None:
        self.settings = settings or Settings()
        self.client = client or AsyncOpenAI(
            api_key=self.settings.token,
            base_url=self.settings.api_url.rstrip("/") + "/v1",
            timeout=self.settings.request_timeout_sec,
            max_retries=0,
        )
        self.mcp_factory = mcp_factory

    async def retrieve(self, messages: list[dict[str, str]]) -> RetrievalResult:
        query = _conversation_context(messages)
        retrieval_messages: list[dict[str, Any]] = [
            {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Use the available MCP tools to collect evidence for the latest user request. "
                    "Resolve references from the full conversation below.\n\n"
                    f"CONVERSATION:\n{query}"
                ),
            },
        ]
        documents: list[str] = []
        action_count = 0

        async with self.mcp_factory(
            self.settings.mcp_url,
            self.settings.token,
            self.settings.request_timeout_sec,
        ) as mcp:
            tools = await mcp.openai_tools()
            if not tools:
                return RetrievalResult(
                    status="no_evidence",
                    note="The MCP server returned no tools.",
                )
            _log("retrieval_started", available_tools=len(tools))

            for round_number in range(1, self.settings.max_retrieval_rounds + 1):
                response = await self.client.chat.completions.create(
                    model=self.settings.model,
                    messages=retrieval_messages,
                    tools=tools,
                    tool_choice="required" if round_number == 1 else "auto",
                    temperature=0,
                    max_tokens=self.settings.retrieval_max_tokens,
                )
                message = response.choices[0].message
                calls = message.tool_calls or []
                if not calls:
                    _log("retrieval_completed", round=round_number, tool_calls=action_count)
                    break

                retrieval_messages.append(_assistant_message(message))
                _log("retrieval_round", round=round_number, requested_calls=len(calls))

                for call in calls:
                    if action_count >= self.settings.max_retrieval_calls:
                        output = "Retrieval tool-call budget exhausted."
                    else:
                        _log(
                            "mcp_call_started",
                            tool=call.function.name,
                            call_number=action_count + 1,
                        )
                        try:
                            output = await mcp.call(
                                call.function.name,
                                _arguments(call.function.arguments),
                            )
                        except Exception as exc:  # noqa: BLE001 - isolate one MCP failure
                            _log(
                                "mcp_call_failed",
                                tool=call.function.name,
                                error_type=type(exc).__name__,
                            )
                            output = f"MCP tool unavailable: {type(exc).__name__}"
                        else:
                            action_count += 1
                            _log(
                                "mcp_call_completed",
                                tool=call.function.name,
                                output_chars=len(output),
                            )
                            if output and not output.startswith("MCP tool error:"):
                                documents.append(
                                    output[: self.settings.max_tool_result_chars]
                                )

                    retrieval_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": output[: self.settings.max_tool_result_chars],
                        }
                    )

                if action_count >= self.settings.max_retrieval_calls:
                    _log("retrieval_budget_exhausted", tool_calls=action_count)
                    break

        evidence = rank_documents(query, documents, self.settings.retrieval_top_k)
        if not evidence:
            return RetrievalResult(
                status="no_evidence",
                note="MCP retrieval returned no citable evidence.",
                tool_calls=action_count,
            )
        return RetrievalResult(
            status="partial",
            evidence=evidence,
            note="MCP retrieval returned the most relevant citable evidence.",
            tool_calls=action_count,
        )

    async def chat(self, messages: list[dict[str, str]]) -> str:
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("messages must end with a user message")

        generation_prompt = (
            GENERATION_SYSTEM_PROMPT
            + "\n"
            + _response_language_instruction(messages[-1]["content"])
        )
        generation_messages: list[dict[str, Any]] = [
            {"role": "system", "content": generation_prompt},
            *messages,
        ]
        first_response = await self.client.chat.completions.create(
            model=self.settings.model,
            messages=generation_messages,
            tools=[RETRIEVE_TOOL],
            tool_choice="auto",
            temperature=0,
            max_tokens=self.settings.generation_max_tokens,
        )
        first_message = first_response.choices[0].message
        calls = first_message.tool_calls or []
        if not calls:
            content = first_message.content or ""
            if not content.strip():
                raise RuntimeError("L2 returned neither an answer nor a retrieval request")
            _log("generation_memory_answer")
            return content

        retrieve_calls = [
            call for call in calls if call.function.name == RETRIEVE_TOOL["function"]["name"]
        ]
        if not retrieve_calls:
            raise RuntimeError("Generation returned an unsupported tool call")

        selected_call = retrieve_calls[0]
        query = _arguments(selected_call.function.arguments).get("query")
        if not isinstance(query, str) or not query.strip():
            raise RuntimeError("Generation returned an empty retrieval query")
        query = query.strip()
        _log("generation_requested_retrieval", query_chars=len(query))

        try:
            async with asyncio.timeout(self.settings.retrieval_timeout_sec):
                retrieval = await self.retrieve([{"role": "user", "content": query}])
        except Exception as exc:  # noqa: BLE001 - retrieval failure must not lose the answer
            _log("retrieval_unavailable", error_type=type(exc).__name__)
            retrieval = RetrievalResult(
                status="no_evidence",
                note="Retrieval was unavailable within the response-time budget.",
            )

        generation_messages.append(_assistant_message(first_message))
        for call in calls:
            content = (
                retrieval.for_generation(self.settings.max_evidence_chars)
                if call.id == selected_call.id
                else "Only one retrieval call is allowed per answer."
            )
            generation_messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )

        final_response = await self.client.chat.completions.create(
            model=self.settings.model,
            messages=generation_messages,
            temperature=0,
            max_tokens=self.settings.generation_max_tokens,
        )
        content = final_response.choices[0].message.content or ""
        if not content.strip():
            raise RuntimeError("L2 returned an empty generation response")
        return content


def _log(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)
