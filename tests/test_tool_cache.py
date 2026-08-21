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


def _response(calls: list[Any]) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=calls)
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
        self, hyde_started: asyncio.Event, connection_started: asyncio.Event
    ) -> None:
        self.hyde_started = hyde_started
        self.connection_started = connection_started

    async def __aenter__(self) -> Self:
        self.connection_started.set()
        await asyncio.wait_for(self.hyde_started.wait(), timeout=0.5)
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def openai_tools(self) -> list[dict[str, Any]]:
        return [_tool("test_tool")]

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        return '{"cite_uid":"cite-test","content":"grounded evidence"}'


def test_hyde_and_mcp_connection_start_concurrently() -> None:
    async def run() -> None:
        hyde_started = asyncio.Event()
        connection_started = asyncio.Event()
        fake_mcp = CoordinatedMCP(hyde_started, connection_started)
        create = AsyncMock(
            side_effect=[
                _response([_call("test_tool", '{"query":"test"}')]),
                _response(
                    [
                        _call(
                            "submit_reflection",
                            '{"sufficient":true,"analysis_summary":"enough",'
                            '"next_query":""}',
                        )
                    ]
                ),
                _response(
                    [
                        _call(
                            "finalize_retrieval",
                            '{"status":"sufficient","items":['
                            '{"cite_uid":"cite-test","relevance_score":0.9}],'
                            '"note":""}',
                        )
                    ]
                ),
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        harness = L2Harness(
            settings=Settings(LUNIT_FM_API_KEY="lunit_test"),
            client=client,  # type: ignore[arg-type]
            mcp_factory=lambda *_: fake_mcp,  # type: ignore[arg-type]
        )

        async def coordinated_hyde(query: str) -> str:
            hyde_started.set()
            await asyncio.wait_for(connection_started.wait(), timeout=0.5)
            return "grounded evidence"

        harness.create_hypothetical_passage = coordinated_hyde  # type: ignore[method-assign]
        result = await asyncio.wait_for(harness.retrieve("test query"), timeout=1)

        assert result.status == "sufficient"
        assert result.evidence[0].cite_uid == "cite-test"

    asyncio.run(run())
