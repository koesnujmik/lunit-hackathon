import json
from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI

from .config import Settings
from .mcp_client import LunitMCPClient
from .models import CitationSelection, Evidence, ReflectionDecision, RetrievalResult
from .prompts import (
    GENERATION_SYSTEM_PROMPT,
    HYDE_SYSTEM_PROMPT,
    REACT_ACTION_SYSTEM_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    TOOL_SELECTOR_SYSTEM_PROMPT,
)
from .ranking import rank_documents, rank_tool_candidates


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
    "Retrieve real evidence. Pass one self-contained query that resolves conversation context.",
    {"query": {"type": "string"}},
)
REFLECTION_TOOL = _decision_tool(
    "submit_reflection",
    "Submit evidence sufficiency and the next retrieval query.",
    {
        "sufficient": {"type": "boolean"},
        "analysis_summary": {"type": "string"},
        "next_query": {"type": "string"},
    },
)
FINALIZE_TOOL = _decision_tool(
    "finalize_retrieval",
    "Select citations and end retrieval.",
    {
        "status": {"type": "string", "enum": ["sufficient", "partial", "no_evidence"]},
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


def _extract_evidence(tool_outputs: list[str], selection: CitationSelection) -> list[Evidence]:
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


def _evidence_context(evidence: list[Evidence]) -> str:
    if not evidence:
        return "No citable evidence yet."
    return "\n\n".join(
        f"[{index}] cite_uid={item.cite_uid} tfidf={item.tfidf_score:.4f}\n{item.content}"
        for index, item in enumerate(evidence, 1)
    )


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
            max_retries=2,
        )
        self.mcp_factory = mcp_factory

    async def _forced_decision(
        self, system_prompt: str, user_content: str, tool: dict[str, Any]
    ) -> dict[str, Any]:
        name = tool["function"]["name"]
        response = await self.client.chat.completions.create(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            tools=[tool],
            tool_choice="required",
            temperature=0,
        )
        calls = response.choices[0].message.tool_calls or []
        if not calls:
            raise RuntimeError(f"L2 did not call required tool {name}")
        return _arguments(calls[0].function.arguments)

    async def create_hypothetical_passage(self, query: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
        )
        return (response.choices[0].message.content or query).strip()

    async def _choose_actions(
        self,
        tools: list[dict[str, Any]],
        query: str,
        passage: str,
        evidence: list[Evidence] | None = None,
        observations: list[str] | None = None,
        reflection: ReflectionDecision | None = None,
    ) -> list[Any]:
        if reflection is None:
            system = TOOL_SELECTOR_SYSTEM_PROMPT
            content = f"QUERY:\n{query}\n\nHYPOTHETICAL PASSAGE:\n{passage}"
        else:
            system = REACT_ACTION_SYSTEM_PROMPT
            recent = "\n\n".join((observations or [])[-3:])[-12000:]
            content = (
                f"QUERY:\n{query}\n\nHYPOTHETICAL PASSAGE:\n{passage}"
                f"\n\nTOP REAL EVIDENCE:\n{_evidence_context(evidence or [])}"
                f"\n\nRECENT TOOL OBSERVATIONS:\n{recent or 'None'}"
                f"\n\nREFLECTION SUMMARY:\n{reflection.analysis_summary}"
                f"\n\nNEXT QUERY:\n{reflection.next_query}"
            )
        response = await self.client.chat.completions.create(
            model=self.settings.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            tools=tools,
            tool_choice="required",
            temperature=0,
        )
        return response.choices[0].message.tool_calls or []

    async def _reflect(
        self, query: str, evidence: list[Evidence], round_number: int
    ) -> ReflectionDecision:
        content = (
            f"QUERY:\n{query}\n\nROUND: {round_number}"
            f"\n\nCURRENT REAL EVIDENCE:\n{_evidence_context(evidence)}"
        )
        arguments = await self._forced_decision(
            REFLECTION_SYSTEM_PROMPT, content, REFLECTION_TOOL
        )
        return ReflectionDecision.model_validate(arguments)

    async def _finalize(
        self, query: str, evidence: list[Evidence], sufficient: bool, note: str
    ) -> RetrievalResult:
        if not evidence:
            return RetrievalResult(status="no_evidence", note=note)
        content = (
            f"Query: {query}\n\nCandidates:\n{_evidence_context(evidence)}\n\n"
            f"Reflection sufficient={sufficient}. Call finalize_retrieval now."
        )
        arguments = await self._forced_decision(
            "Select final cite_uids only from supplied real evidence. Do not answer.",
            content,
            FINALIZE_TOOL,
        )
        selection = CitationSelection.model_validate(arguments)
        allowed = {item.cite_uid: item for item in evidence}
        selected = [allowed[item.cite_uid] for item in selection.items if item.cite_uid in allowed]
        return RetrievalResult(
            status=selection.status if selected else "no_evidence",
            evidence=selected,
            note=selection.note or note,
        )

    async def retrieve(self, query: str) -> RetrievalResult:
        passage = await self.create_hypothetical_passage(query)
        documents: list[str] = []
        action_count = 0
        evidence: list[Evidence] = []
        reflection = ReflectionDecision(
            sufficient=False,
            analysis_summary="Initial selection has not run.",
            next_query=query,
        )
        async with self.mcp_factory(
            self.settings.mcp_url, self.settings.token, self.settings.request_timeout_sec
        ) as mcp:
            tools = await mcp.openai_tools()
            candidate_tools = rank_tool_candidates(
                f"{query}\n{passage}", tools, self.settings.tool_candidate_limit
            )
            actions = await self._choose_actions(candidate_tools, query, passage)
            for round_number in range(1, self.settings.max_reflection_rounds + 1):
                for action in actions:
                    if action_count >= self.settings.max_retrieval_calls:
                        break
                    output = await mcp.call(action.function.name, _arguments(action.function.arguments))
                    documents.append(output)
                    action_count += 1
                evidence = rank_documents(passage, documents, self.settings.retrieval_top_k)
                reflection = await self._reflect(query, evidence, round_number)
                if reflection.sufficient or action_count >= self.settings.max_retrieval_calls:
                    break
                actions = await self._choose_actions(
                    candidate_tools,
                    query,
                    passage,
                    evidence=evidence,
                    observations=documents,
                    reflection=reflection,
                )
        result = await self._finalize(
            query, evidence, reflection.sufficient, reflection.analysis_summary
        )
        result.tool_calls = action_count
        return result

    async def chat(self, messages: list[dict[str, str]]) -> str:
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("messages must end with a user message")
        generation_prompt = (
            GENERATION_SYSTEM_PROMPT
            + "\n"
            + _response_language_instruction(messages[-1]["content"])
        )
        generation_messages: list[dict[str, Any]] = [
            {"role": "system", "content": generation_prompt}, *messages
        ]
        response = await self.client.chat.completions.create(
            model=self.settings.model,
            messages=generation_messages,
            tools=[RETRIEVE_TOOL],
            tool_choice="auto",
            temperature=0.2,
        )
        message = response.choices[0].message
        calls = message.tool_calls or []
        if not calls:
            return message.content or ""
        retrieve_calls = [
            call for call in calls if call.function.name == "retrieve_relevant_content"
        ]
        if not retrieve_calls:
            raise RuntimeError("Generation returned an unsupported tool call")

        selected_call = retrieve_calls[0]
        query = _arguments(selected_call.function.arguments)["query"]
        result = await self.retrieve(query)
        generation_messages.append(_assistant_message(message))
        for call in calls:
            content = (
                result.for_generation()
                if call.id == selected_call.id
                else "Only one retrieval call is allowed per answer."
            )
            generation_messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )
        final_response = await self.client.chat.completions.create(
            model=self.settings.model,
            messages=generation_messages,
            temperature=0.2,
        )
        return final_response.choices[0].message.content or ""
