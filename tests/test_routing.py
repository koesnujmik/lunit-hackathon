import asyncio
import json
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import AsyncMock, Mock

from httpx import Request, Response
from openai import InternalServerError

from l2_baseline.config import Settings
from l2_baseline.harness import (
    L2Harness,
    _compact_messages,
    _deterministic_guideline_request,
    _index_page_arguments,
    _needs_retrieval,
    _prepare_primary_arguments,
    _response_language_instruction,
)


def _tool(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }


def _response(
    content: str = "",
    tool_calls: list[object] | None = None,
    finish_reason: str | None = None,
) -> object:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


def _call(name: str, arguments: str, call_id: str = "call-1") -> Mock:
    call = Mock()
    call.id = call_id
    call.function.name = name
    call.function.arguments = arguments
    return call


class FakeMCP:
    def __init__(self) -> None:
        self.tools = [
            _tool("index_get_relevant_nodes", "Find relevant clinical guideline sections"),
            _tool("openapi_law_search", "Search Korean laws"),
            _tool("kcd_search_codes", "Search KCD disease codes"),
        ]
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def openai_tools(self) -> list[dict[str, Any]]:
        return self.tools

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        if name == "index_get_relevant_nodes":
            return (
                '[{"doc_id":"ckd-guide","range":[12,14],"title":"Blood pressure '
                'targets","summary":"CKD blood pressure target recommendations",'
                '"score":0.9}]'
            )
        return '{"cite_uid":"cite-1","content":"retrieved guideline evidence"}'


def _harness(
    responses: list[object], mcp: FakeMCP | None = None
) -> tuple[L2Harness, AsyncMock, FakeMCP]:
    create = AsyncMock(side_effect=responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    settings = Settings(LUNIT_FM_API_KEY="lunit_test", _env_file=None)
    fake_mcp = mcp or FakeMCP()
    harness = L2Harness(
        settings=settings,
        client=client,  # type: ignore[arg-type]
        mcp_factory=lambda *_args: fake_mcp,  # type: ignore[arg-type]
    )
    return harness, create, fake_mcp


def test_response_language_is_explicit() -> None:
    assert "English only" in _response_language_instruction("My knee clicks")
    assert "Korean" in _response_language_instruction("무릎에서 소리가 나요")


def test_router_only_retrieves_source_specific_requests() -> None:
    assert not _needs_retrieval(
        [{"role": "user", "content": "What commonly causes a sore throat?"}]
    )
    assert _needs_retrieval(
        [
            {
                "role": "user",
                "content": "According to current clinical guidelines, what is the BP target?",
            }
        ]
    )
    assert _needs_retrieval(
        [{"role": "user", "content": "이 약의 식약처 허가사항과 급여 기준을 알려줘"}]
    )


def test_compaction_preserves_first_user_request_and_recent_turns() -> None:
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"message-{index}"}
        for index in range(11)
    ]

    compact = _compact_messages(messages, max_messages=6, max_chars=4_000)

    assert len(compact) == 6
    assert compact[0] == messages[0]
    assert compact[1:] == messages[-5:]


def test_index_search_is_bounded_and_expanded() -> None:
    prepared = _prepare_primary_arguments(
        "index_get_relevant_nodes",
        {"corpus_tag": "guideline", "query": "CKD blood pressure", "k": 50},
    )

    assert prepared["k"] == 8
    assert "exact recommendation" in prepared["query"]


def test_unambiguous_guideline_route_skips_tool_selector() -> None:
    request = _deterministic_guideline_request(
        "According to current clinical guidelines, what is the CKD target?",
        {"index_get_relevant_nodes"},
    )

    assert request is not None
    assert request[0] == "index_get_relevant_nodes"
    assert request[1]["corpus_tag"] == "guideline"
    assert _deterministic_guideline_request(
        "심평원 급여 기준 가이드라인", {"index_get_relevant_nodes"}
    ) is None


def test_index_followup_prefers_answer_bearing_population_match() -> None:
    nodes = [
        {
            "doc_id": "scope-only",
            "range": [3, 3],
            "title": "Chronic kidney disease",
            "summary": "Blood pressure management is not addressed in this guideline.",
            "score": 0.95,
        },
        {
            "doc_id": "answer-node",
            "range": [20, 24],
            "title": "Blood pressure goal",
            "summary": (
                "For adults with chronic kidney disease, the recommended blood pressure "
                "target is described in this section."
            ),
            "score": 0.8,
        },
    ]

    arguments = _index_page_arguments(
        "recommended blood pressure target adults chronic kidney disease",
        {"corpus_tag": "guideline"},
        json.dumps(nodes),
    )

    assert arguments == {
        "corpus_tag": "guideline",
        "doc_id": "answer-node",
        "start_page": 20,
        "end_page": 23,
    }


def test_index_followup_honors_current_source_request() -> None:
    nodes = [
        {
            "doc_id": "older",
            "range": [10, 10],
            "doc_title": "2019 clinical guideline",
            "summary": "The exact blood pressure target for chronic kidney disease.",
            "score": 0.9,
        },
        {
            "doc_id": "newer",
            "range": [20, 20],
            "doc_title": "2025 clinical guideline",
            "summary": "The blood pressure target for adults with chronic kidney disease.",
            "score": 0.85,
        },
    ]

    arguments = _index_page_arguments(
        "current blood pressure target adults chronic kidney disease",
        {"corpus_tag": "guideline"},
        json.dumps(nodes),
    )

    assert arguments is not None
    assert arguments["doc_id"] == "newer"


def test_direct_route_uses_one_l2_call() -> None:
    harness, create, mcp = _harness([_response(content="direct answer")])

    answer = asyncio.run(
        harness.chat([{"role": "user", "content": "What commonly causes a sore throat?"}])
    )

    assert answer == "direct answer"
    assert create.await_count == 1
    assert "tools" not in create.await_args.kwargs
    assert mcp.calls == []


def test_empty_direct_answer_is_retried_once_locally() -> None:
    harness, create, _mcp = _harness(
        [_response(content=""), _response(content="recovered answer")]
    )

    answer = asyncio.run(harness.chat([{"role": "user", "content": "general question"}]))

    assert answer == "recovered answer"
    assert create.await_count == 2
    retry_request = create.await_args_list[1].kwargs
    assert "RECOVERY INSTRUCTION" in retry_request["messages"][0]["content"]
    assert retry_request["max_tokens"] == 512


def test_length_only_response_gets_larger_recovery_budget() -> None:
    harness, create, _mcp = _harness(
        [
            _response(content="", finish_reason="length"),
            _response(content="recovered answer"),
        ]
    )

    answer = asyncio.run(harness.chat([{"role": "user", "content": "complex question"}]))

    assert answer == "recovered answer"
    assert create.await_args_list[1].kwargs["max_tokens"] == 2_048


def test_empty_then_transient_500_can_recover_within_local_budget() -> None:
    upstream_error = InternalServerError(
        "temporary failure",
        response=Response(500, request=Request("POST", "https://model.test")),
        body=None,
    )
    harness, create, _mcp = _harness(
        [
            _response(content=""),
            upstream_error,
            _response(content="recovered answer"),
        ]
    )

    answer = asyncio.run(harness.chat([{"role": "user", "content": "general question"}]))

    assert answer == "recovered answer"
    assert create.await_count == 3


def test_source_route_opens_one_page_then_runs_final_generation() -> None:
    harness, create, mcp = _harness([_response(content="grounded answer")])

    answer = asyncio.run(
        harness.chat(
            [
                {
                    "role": "user",
                    "content": "According to current clinical guidelines, what is the CKD BP target?",
                }
            ]
        )
    )

    assert answer == "grounded answer"
    assert create.await_count == 1
    assert len(mcp.calls) == 2
    assert [name for name, _arguments in mcp.calls] == [
        "index_get_relevant_nodes",
        "index_get_page_content",
    ]
    assert mcp.calls[0][1]["corpus_tag"] == "guideline"
    assert mcp.calls[0][1]["k"] == 8
    assert "CKD BP target" in mcp.calls[0][1]["query"]
    assert mcp.calls[1][1] == {
        "corpus_tag": "guideline",
        "doc_id": "ckd-guide",
        "start_page": 12,
        "end_page": 14,
    }
    final_request = create.await_args_list[0].kwargs
    assert "tools" not in final_request
    assert "cite-1" in final_request["messages"][0]["content"]


def test_structured_source_route_uses_one_l2_tool_selection() -> None:
    harness, create, mcp = _harness(
        [
            _response(tool_calls=[_call("kcd_search_codes", '{"query":"당뇨병"}')]),
            _response(content="E11 [1]"),
        ]
    )

    answer = asyncio.run(
        harness.chat([{"role": "user", "content": "당뇨병의 정확한 KCD 코드는?"}])
    )

    assert answer == "E11 [1]"
    assert create.await_count == 2
    assert create.await_args_list[0].kwargs["tool_choice"] == "required"
    assert mcp.calls == [("kcd_search_codes", {"query": "당뇨병"})]


def test_retrieval_failure_still_runs_final_l2_generation() -> None:
    harness, create, _mcp = _harness([_response(content="safe answer with limitation")])
    harness.retrieve = AsyncMock(side_effect=TimeoutError)

    answer = asyncio.run(
        harness.chat(
            [{"role": "user", "content": "According to the guideline, what is recommended?"}]
        )
    )

    assert answer == "safe answer with limitation"
    assert create.await_count == 1
    assert "Retrieval was unavailable" in create.await_args.kwargs["messages"][0]["content"]
