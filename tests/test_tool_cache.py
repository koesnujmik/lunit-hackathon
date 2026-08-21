import asyncio
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import AsyncMock, Mock

from l2_baseline.config import Settings
from l2_baseline.harness import L2Harness


def _tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


def _call(name: str, arguments: str) -> Mock:
    call = Mock()
    call.function.name = name
    call.function.arguments = arguments
    return call


def _response(calls: list[Any], content: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=calls)
            )
        ]
    )


class CountingMCP:
    def __init__(self) -> None:
        self.list_calls = 0

    async def openai_tools(self) -> list[dict[str, Any]]:
        self.list_calls += 1
        await asyncio.sleep(0)
        return [_tool("test_tool")]


def test_tool_schemas_are_cached_across_concurrent_requests() -> None:
    harness = L2Harness(
        settings=Settings(LUNIT_FM_API_KEY="lunit_test"),
        client=SimpleNamespace(),  # type: ignore[arg-type]
    )
    mcp = CountingMCP()

    async def run() -> None:
        first, second = await asyncio.gather(
            harness._get_mcp_tools(mcp),  # type: ignore[arg-type]
            harness._get_mcp_tools(mcp),  # type: ignore[arg-type]
        )
        third = await harness._get_mcp_tools(mcp)  # type: ignore[arg-type]
        assert first == second == third

    asyncio.run(run())

    assert mcp.list_calls == 1


class CoordinatedMCP:
    def __init__(
        self, assessment_started: asyncio.Event, connection_started: asyncio.Event
    ) -> None:
        self.assessment_started = assessment_started
        self.connection_started = connection_started

    async def __aenter__(self) -> Self:
        self.connection_started.set()
        await asyncio.wait_for(self.assessment_started.wait(), timeout=0.5)
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def openai_tools(self) -> list[dict[str, Any]]:
        return [_tool("test_tool")]

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        return '{"cite_uid":"cite-test","content":"grounded evidence"}'


def test_query_assessment_and_mcp_connection_start_concurrently() -> None:
    async def run() -> None:
        assessment_started = asyncio.Event()
        connection_started = asyncio.Event()
        fake_mcp = CoordinatedMCP(assessment_started, connection_started)

        async def create_response(**kwargs: Any) -> SimpleNamespace:
            tools = kwargs.get("tools") or []
            tool_names = [tool["function"]["name"] for tool in tools]
            if "submit_query_assessment" in tool_names:
                assessment_started.set()
                await asyncio.wait_for(connection_started.wait(), timeout=0.5)
                return _response(
                    [
                        _call(
                            "submit_query_assessment",
                            '{"query_sufficient":true,"reason":"specific query"}',
                        )
                    ]
                )
            if tool_names == ["test_tool"]:
                return _response([_call("test_tool", '{"query":"test"}')])
            if "finalize_retrieval" in tool_names:
                return _response(
                    [
                        _call(
                            "finalize_retrieval",
                            '{"status":"sufficient","items":['
                            '{"cite_uid":"cite-test","relevance_score":0.9}],'
                            '"note":""}',
                        )
                    ]
                )
            raise AssertionError(
                f"query-sufficient retrieval made an unexpected L2 call: {tool_names}"
            )

        create = AsyncMock(side_effect=create_response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        harness = L2Harness(
            settings=Settings(LUNIT_FM_API_KEY="lunit_test"),
            client=client,  # type: ignore[arg-type]
            mcp_factory=lambda *_: fake_mcp,  # type: ignore[arg-type]
        )
        result = await asyncio.wait_for(harness.retrieve("test query"), timeout=1)

        assert result.status == "sufficient"
        assert result.evidence[0].cite_uid == "cite-test"
        assert assessment_started.is_set()
        assert connection_started.is_set()

    asyncio.run(run())


def test_query_assessment_generates_rationale_only_when_needed() -> None:
    async def run() -> None:
        assessment_started = asyncio.Event()
        connection_started = asyncio.Event()
        fake_mcp = CoordinatedMCP(assessment_started, connection_started)
        rationale = "official label interaction evidence and monitoring requirements"
        selector_contents: list[str] = []

        async def create_response(**kwargs: Any) -> SimpleNamespace:
            tools = kwargs.get("tools") or []
            tool_names = [tool["function"]["name"] for tool in tools]
            if "submit_query_assessment" in tool_names:
                assessment_started.set()
                await asyncio.wait_for(connection_started.wait(), timeout=0.5)
                return _response(
                    [
                        _call(
                            "submit_query_assessment",
                            '{"query_sufficient":false,"reason":"expansion needed"}',
                        )
                    ]
                )
            if not tools:
                return _response([], content=rationale)
            if tool_names == ["test_tool"]:
                selector_contents.append(kwargs["messages"][-1]["content"])
                return _response([_call("test_tool", '{"query":"test"}')])
            if "finalize_retrieval" in tool_names:
                return _response(
                    [
                        _call(
                            "finalize_retrieval",
                            '{"status":"sufficient","items":['
                            '{"cite_uid":"cite-test","relevance_score":0.9}],'
                            '"note":""}',
                        )
                    ]
                )
            raise AssertionError(f"unexpected L2 call: {tool_names}")

        create = AsyncMock(side_effect=create_response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        harness = L2Harness(
            settings=Settings(LUNIT_FM_API_KEY="lunit_test"),
            client=client,  # type: ignore[arg-type]
            mcp_factory=lambda *_: fake_mcp,  # type: ignore[arg-type]
        )

        result = await asyncio.wait_for(harness.retrieve("ambiguous query"), timeout=1)

        assert result.status == "sufficient"
        assert selector_contents and rationale in selector_contents[0]
        assert any(not call.kwargs.get("tools") for call in create.await_args_list)

    asyncio.run(run())
