import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI

from .config import Settings
from .mcp_client import LunitMCPClient
from .models import CitationSelection, Evidence, RetrievalQueryDecision, RetrievalResult
from .prompts import (
    GENERATION_SYSTEM_PROMPT,
    QUERY_ASSESSMENT_SYSTEM_PROMPT,
    RETRIEVAL_RATIONALE_SYSTEM_PROMPT,
    TOOL_SELECTOR_SYSTEM_PROMPT,
)
from .ranking import rank_documents, rank_tool_candidates

logger = logging.getLogger(__name__)


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
QUERY_ASSESSMENT_TOOL = _decision_tool(
    "submit_query_assessment",
    "Decide whether the query alone is sufficient for retrieval and ranking.",
    {
        "query_sufficient": {"type": "boolean"},
        "reason": {"type": "string"},
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
        f"[{index}] cite_uid={item.cite_uid} bm25={item.bm25_score:.4f}\n{item.content}"
        for index, item in enumerate(evidence, 1)
    )


async def _run_mcp_actions(
    mcp: LunitMCPClient, actions: list[Any], remaining_budget: int
) -> list[str]:
    """Run one MCP batch concurrently and isolate failures by action."""
    selected_actions = actions[: max(remaining_budget, 0)]

    async def run(action: Any) -> str:
        started = time.perf_counter()
        try:
            return await mcp.call(
                action.function.name, _arguments(action.function.arguments)
            )
        except Exception:
            logger.warning(
                "MCP action failed tool=%s",
                action.function.name,
                exc_info=True,
            )
            return ""
        finally:
            logger.info(
                "timing stage=mcp_tool tool=%s duration_sec=%.3f",
                action.function.name,
                time.perf_counter() - started,
            )

    calls = [run(action) for action in selected_actions]
    if not calls:
        return []
    return list(await asyncio.gather(*calls))


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
        self._mcp_tools_cache: list[dict[str, Any]] | None = None
        self._mcp_tools_lock = asyncio.Lock()

    async def _get_mcp_tools(self, mcp: LunitMCPClient) -> list[dict[str, Any]]:
        """Cache the static MCP tool schemas for this process-level harness."""
        started = time.perf_counter()
        loaded = False
        try:
            if self._mcp_tools_cache is None:
                async with self._mcp_tools_lock:
                    if self._mcp_tools_cache is None:
                        self._mcp_tools_cache = await mcp.openai_tools()
                        loaded = True
            return self._mcp_tools_cache
        finally:
            logger.info(
                "timing stage=mcp_list_tools cache_hit=%s duration_sec=%.3f",
                not loaded,
                time.perf_counter() - started,
            )

    async def _forced_decision(
        self, system_prompt: str, user_content: str, tool: dict[str, Any]
    ) -> dict[str, Any]:
        name = tool["function"]["name"]
        started = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                tools=[tool],
                tool_choice="auto",
                temperature=0,
            )
        finally:
            logger.info(
                "timing stage=l2_forced_decision tool=%s duration_sec=%.3f",
                name,
                time.perf_counter() - started,
            )
        calls = response.choices[0].message.tool_calls or []
        if not calls:
            raise RuntimeError(f"L2 did not call required tool {name}")
        return _arguments(calls[0].function.arguments)

    async def _assess_retrieval_query(self, query: str) -> RetrievalQueryDecision:
        arguments = await self._forced_decision(
            QUERY_ASSESSMENT_SYSTEM_PROMPT,
            f"QUERY:\n{query}",
            QUERY_ASSESSMENT_TOOL,
        )
        return RetrievalQueryDecision.model_validate(arguments)

    async def create_retrieval_rationale(self, query: str) -> str:
        started = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.model,
                messages=[
                    {"role": "system", "content": RETRIEVAL_RATIONALE_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.2,
            )
        finally:
            logger.info(
                "timing stage=retrieval_rationale duration_sec=%.3f",
                time.perf_counter() - started,
            )
        return (response.choices[0].message.content or query).strip()

    async def _choose_actions(
        self,
        tools: list[dict[str, Any]],
        query: str,
        rationale: str = "",
    ) -> list[Any]:
        content = f"QUERY:\n{query}"
        if rationale:
            content += f"\n\nRETRIEVAL RATIONALE:\n{rationale}"
        else:
            content += "\n\nRETRIEVAL RATIONALE:\nNot required; use the query alone."
        started = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.model,
                messages=[
                    {"role": "system", "content": TOOL_SELECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                tools=tools,
                tool_choice="auto",
                temperature=0,
            )
        finally:
            logger.info(
                "timing stage=tool_selection duration_sec=%.3f",
                time.perf_counter() - started,
            )
        return response.choices[0].message.tool_calls or []

    async def _finalize(
        self, query: str, evidence: list[Evidence], note: str
    ) -> RetrievalResult:
        if not evidence:
            return RetrievalResult(status="no_evidence", note=note)
        content = (
            f"Query: {query}\n\nCandidates:\n{_evidence_context(evidence)}\n\n"
            "The single-pass retrieval is complete. Determine evidence sufficiency and call "
            "finalize_retrieval now."
        )
        arguments = await self._forced_decision(
            "Select final cite_uids only from supplied real evidence, determine whether it is "
            "sufficient, partial, or absent, and do not answer the medical query.",
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
        retrieval_started = time.perf_counter()
        assessment_task = asyncio.create_task(self._assess_retrieval_query(query))
        documents: list[str] = []
        action_count = 0
        evidence: list[Evidence] = []
        decision: RetrievalQueryDecision | None = None
        rationale = ""
        try:
            mcp_connect_started = time.perf_counter()
            async with self.mcp_factory(
                self.settings.mcp_url,
                self.settings.token,
                self.settings.request_timeout_sec,
            ) as mcp:
                logger.info(
                    "timing stage=mcp_connect duration_sec=%.3f",
                    time.perf_counter() - mcp_connect_started,
                )
                tools = await self._get_mcp_tools(mcp)
                decision = await assessment_task
                if not decision.query_sufficient:
                    rationale = await self.create_retrieval_rationale(query)
                candidate_tools = rank_tool_candidates(
                    query,
                    tools,
                    self.settings.tool_candidate_limit,
                    rationale=rationale,
                )
                actions = await self._choose_actions(candidate_tools, query, rationale)
                outputs = await _run_mcp_actions(
                    mcp, actions, self.settings.max_retrieval_calls
                )
                documents.extend(outputs)
                action_count = len(outputs)
                evidence = rank_documents(query, rationale, documents)
        finally:
            if not assessment_task.done():
                assessment_task.cancel()
            await asyncio.gather(assessment_task, return_exceptions=True)

        strategy = "query-only" if decision and decision.query_sufficient else "query+rationale"
        note = f"Single-pass {strategy} retrieval."
        if decision and decision.reason:
            note += f" Query assessment: {decision.reason}"
        result = await self._finalize(query, evidence, note)
        result.tool_calls = action_count
        logger.info(
            "timing stage=retrieval_total tool_calls=%d duration_sec=%.3f",
            action_count,
            time.perf_counter() - retrieval_started,
        )
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
        routing_started = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.model,
                messages=generation_messages,
                tools=[RETRIEVE_TOOL],
                tool_choice="auto",
                temperature=0.2,
            )
        finally:
            logger.info(
                "timing stage=generation_routing duration_sec=%.3f",
                time.perf_counter() - routing_started,
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
        final_started = time.perf_counter()
        try:
            final_response = await self.client.chat.completions.create(
                model=self.settings.model,
                messages=generation_messages,
                temperature=0.2,
            )
        finally:
            logger.info(
                "timing stage=final_generation duration_sec=%.3f",
                time.perf_counter() - final_started,
            )
        return final_response.choices[0].message.content or ""
