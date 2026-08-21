import asyncio
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import AsyncMock, Mock

from l2_baseline.config import Settings
from l2_baseline.harness import RETRIEVE_TOOL, L2Harness, _response_language_instruction


def _tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Description for {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _response(content: str = "", tool_calls: list[object] | None = None) -> object:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _call(name: str, arguments: str, call_id: str = "call-1") -> Mock:
    call = Mock()
    call.id = call_id
    call.function.name = name
    call.function.arguments = arguments
    call.model_dump.return_value = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    return call


class FakeMCP:
    def __init__(self) -> None:
        self.tools = [_tool(f"actual_lunit_tool_{index}") for index in range(21)]
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def openai_tools(self) -> list[dict[str, Any]]:
        return self.tools

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return '{"cite_uid":"cite-1","content":"retrieved evidence"}'


def _harness(
    responses: list[object], mcp: FakeMCP | None = None
) -> tuple[L2Harness, AsyncMock]:
    create = AsyncMock(side_effect=responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    settings = Settings(LUNIT_FM_API_KEY="lunit_test", _env_file=None)
    mcp = mcp or FakeMCP()
    harness = L2Harness(
        settings=settings,
        client=client,  # type: ignore[arg-type]
        mcp_factory=lambda *_args: mcp,  # type: ignore[arg-type]
    )
    return harness, create


def test_response_language_is_explicit() -> None:
    assert "English only" in _response_language_instruction("My knee clicks")
    assert "Korean" in _response_language_instruction("무릎에서 소리가 나요")


def test_memory_answer_skips_retrieval() -> None:
    harness, create = _harness([_response(content="direct memory answer")])
    harness.retrieve = AsyncMock()

    answer = asyncio.run(harness.chat([{"role": "user", "content": "general question"}]))

    assert answer == "direct memory answer"
    harness.retrieve.assert_not_awaited()
    assert create.await_count == 1
    initial_generation = create.await_args.kwargs
    assert initial_generation["tools"] == [RETRIEVE_TOOL]
    assert initial_generation["tool_choice"] == "auto"


def test_retrieval_uses_all_real_mcp_tools_then_returns_to_generation() -> None:
    bridge_call = _call(
        "retrieve_relevant_content",
        '{"query":"self-contained guideline question"}',
        "bridge-call",
    )
    mcp_call = _call(
        "actual_lunit_tool_7",
        '{"query":"self-contained guideline question"}',
        "mcp-call",
    )
    mcp = FakeMCP()
    harness, create = _harness(
        [
            _response(tool_calls=[bridge_call]),
            _response(tool_calls=[mcp_call]),
            _response(content="retrieval complete"),
            _response(content="grounded answer"),
        ],
        mcp,
    )

    answer = asyncio.run(harness.chat([{"role": "user", "content": "guideline question"}]))

    assert answer == "grounded answer"
    assert create.await_count == 4

    initial_generation = create.await_args_list[0].kwargs
    assert initial_generation["tools"] == [RETRIEVE_TOOL]
    assert initial_generation["tool_choice"] == "auto"

    first_retrieval = create.await_args_list[1].kwargs
    assert first_retrieval["tools"] == mcp.tools
    assert len(first_retrieval["tools"]) == 21
    assert first_retrieval["tool_choice"] == "required"

    second_retrieval = create.await_args_list[2].kwargs
    assert second_retrieval["tools"] == mcp.tools
    assert second_retrieval["tool_choice"] == "auto"
    assert second_retrieval["messages"][-2]["tool_calls"][0]["id"] == "mcp-call"
    assert second_retrieval["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "mcp-call",
        "content": '{"cite_uid":"cite-1","content":"retrieved evidence"}',
    }
    assert mcp.calls == [
        ("actual_lunit_tool_7", {"query": "self-contained guideline question"})
    ]

    final_generation = create.await_args_list[3].kwargs
    assert "tools" not in final_generation
    assert final_generation["messages"][-2]["tool_calls"][0]["id"] == "bridge-call"
    assert final_generation["messages"][-1]["tool_call_id"] == "bridge-call"
    assert "cite-1" in final_generation["messages"][-1]["content"]


def test_retrieval_failure_still_returns_a_generation() -> None:
    bridge_call = _call(
        "retrieve_relevant_content",
        '{"query":"current external evidence"}',
        "bridge-call",
    )
    harness, create = _harness(
        [_response(tool_calls=[bridge_call]), _response(content="safe fallback answer")]
    )
    harness.retrieve = AsyncMock(side_effect=TimeoutError)

    answer = asyncio.run(harness.chat([{"role": "user", "content": "guideline question"}]))

    assert answer == "safe fallback answer"
    assert create.await_count == 2
    assert "No citable evidence" in create.await_args.kwargs["messages"][-1]["content"]
