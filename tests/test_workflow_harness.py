import asyncio
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import AsyncMock, Mock

from l2_baseline.config import Settings
from l2_baseline.harness import L2Harness


def _tool(name: str, required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in required},
                "required": required,
            },
        },
    }


def _response(content: str = "", calls: list[Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))
        ]
    )


def _call(name: str, arguments: str) -> Mock:
    call = Mock()
    call.function.name = name
    call.function.arguments = arguments
    return call


class FakeMCP:
    def __init__(self) -> None:
        self.called: list[str] = []
        self.tools = [
            _tool("index_list_documents", ["corpus_tag", "query"]),
            _tool("index_get_relevant_nodes", ["corpus_tag", "query", "node_id"]),
            _tool(
                "index_get_page_content",
                ["corpus_tag", "doc_id", "start_page", "end_page"],
            ),
        ]

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def openai_tools(self) -> list[dict[str, Any]]:
        return self.tools

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        self.called.append(name)
        if name == "index_list_documents":
            return '{"documents":[{"node_id":"node-1","title":"CKD guideline"}]}'
        if name == "index_get_relevant_nodes":
            return '{"nodes":[{"doc_id":"doc-1","start_page":48,"end_page":52}]}'
        return (
            '{"cite_uid":"cite-page","title":"CKD guideline",'
            '"content":"systolic blood pressure target"}'
        )


def test_index_dependency_chain_auto_advances_without_intermediate_reflections() -> None:
    selector_call = _call(
        "index_list_documents",
        '{"corpus_tag":"guideline","query":"CKD blood pressure target"}',
    )
    reflection_call = _call(
        "submit_reflection",
        '{"sufficient":true,"analysis_summary":"Direct page evidence found",'
        '"next_query":""}',
    )
    finalize_call = _call(
        "finalize_retrieval",
        '{"status":"sufficient","items":[{"cite_uid":"cite-page",'
        '"relevance_score":0.95}],"note":""}',
    )
    create = AsyncMock(
        side_effect=[
            _response(content="Need CKD target and guideline evidence."),
            _response(calls=[selector_call]),
            _response(calls=[reflection_call]),
            _response(calls=[finalize_call]),
        ]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    fake_mcp = FakeMCP()
    harness = L2Harness(
        settings=Settings(LUNIT_FM_API_KEY="lunit_test"),
        client=client,  # type: ignore[arg-type]
        mcp_factory=lambda *_: fake_mcp,  # type: ignore[arg-type]
    )

    result = asyncio.run(harness.retrieve("CKD blood pressure target"))

    assert fake_mcp.called == [
        "index_list_documents",
        "index_get_relevant_nodes",
        "index_get_page_content",
    ]
    assert result.status == "sufficient"
    assert result.tool_calls == 3
    assert result.evidence[0].cite_uid == "cite-page"
    assert create.await_count == 4
