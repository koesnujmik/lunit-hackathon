from __future__ import annotations

import unittest
from typing import Any

from lunit_hackathon.api import APIError
from lunit_hackathon.config import Settings
from lunit_hackathon.harness import L2Harness
from lunit_hackathon.mcp import MCPTool
from lunit_hackathon.retrieval import (
    RetrievalEngine,
    RetrievalResult,
    _select_mcp_tools,
    _serialize_evidence,
)
from lunit_hackathon.routing import explicitly_requests_retrieval, source_families


def settings() -> Settings:
    return Settings(
        fm_api_url="https://model.example",
        fm_api_key="test-key",
        fm_model="Lunit/L2-preview",
        mcp_url="https://mcp.example/mcp",
        patient_api_url="https://patient.example",
        patient_model="patient-simulator-ko",
        timeout_sec=1,
        enable_retrieval=True,
        routing_mode="hybrid",
        max_retrieval_tool_calls=3,
        max_tool_result_chars=10_000,
        max_evidence_chars=10_000,
        generation_max_tokens=1_200,
        retrieval_max_tokens=768,
    )


class FakeChatClient:
    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def chat_completions(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str) -> RetrievalResult:
        self.queries.append(query)
        return RetrievalResult(
            "sufficient",
            note="test evidence",
            mcp_tool_calls=2,
        )


class FakeMCPClient:
    def list_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                "source_tool",
                "Fetch a citable source.",
                {"type": "object", "properties": {}, "additionalProperties": False},
            )
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.last_call = (name, arguments)
        return {
            "structuredContent": {
                "cite_uid": "cite-1",
                "title": "Test guideline",
                "content": "Target statement",
            },
            "isError": False,
        }


class FakeIndexMCPClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_tools(self) -> list[MCPTool]:
        schema = {"type": "object", "properties": {}, "additionalProperties": True}
        return [
            MCPTool("index_get_relevant_nodes", "", schema),
            MCPTool("index_get_page_content", "", schema),
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        if name == "index_get_relevant_nodes":
            return {
                "structuredContent": {
                    "result": [
                        {
                            "doc_id": "doc-1",
                            "range": [70, 71],
                            "title": "Medication use in CKD",
                        },
                        {
                            "doc_id": "doc-goal",
                            "range": [48, 55],
                            "title": "BP Goal for Hypertension",
                            "summary": "Recommended blood pressure target",
                        },
                    ]
                }
            }
        return {
            "structuredContent": {
                "cite_uid": "cite-page",
                "title": "Test guideline",
                "content": "Target statement from page content",
            }
        }


class HarnessTests(unittest.TestCase):
    def test_generation_api_error_retries_with_compact_direct_messages(self) -> None:
        client = FakeChatClient(
            [
                APIError("timed out"),
                {"choices": [{"message": {"role": "assistant", "content": "복구 답변"}}]},
            ]
        )
        harness = L2Harness(settings(), chat_client=client, retriever=FakeRetriever())

        answer = harness.generate([{"role": "user", "content": "감기 관리 방법은?"}])

        self.assertEqual(answer, "복구 답변")
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(client.requests[1]["extra_body"], {})
        self.assertLess(len(str(client.requests[1]["messages"])), 10_000)

    def test_source_routing_does_not_treat_generic_recommendation_as_guideline(self) -> None:
        self.assertEqual(source_families("가벼운 감기 관리 방법과 일반적인 권고사항"), set())
        self.assertEqual(source_families("clinical guideline recommendation"), {"guideline"})

    def test_explicit_source_request_forces_retrieval(self) -> None:
        conversation = [{"role": "user", "content": "가이드라인 근거를 알려주세요."}]
        self.assertTrue(explicitly_requests_retrieval(conversation))

    def test_guideline_query_limits_retrieval_tools_to_index_family(self) -> None:
        tools = [
            MCPTool("index_get_relevant_nodes", "", {}),
            MCPTool("index_get_page_content", "", {}),
            MCPTool("openapi_law_search", "", {}),
            MCPTool("rag_vector_query", "", {}),
        ]

        selected = _select_mcp_tools("clinical guideline recommendation with evidence", tools)

        self.assertEqual(
            [tool.name for tool in selected],
            ["index_get_relevant_nodes", "index_get_page_content"],
        )

    def test_direct_generation_uses_one_model_call(self) -> None:
        client = FakeChatClient(
            [{"choices": [{"message": {"role": "assistant", "content": "직접 답변"}}]}]
        )
        harness = L2Harness(settings(), chat_client=client, retriever=FakeRetriever())

        answer = harness.generate([{"role": "user", "content": "감기란 무엇인가요?"}])

        self.assertEqual(answer, "직접 답변")
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0]["extra_body"], {})

    def test_generation_tool_call_runs_retrieval_then_answers(self) -> None:
        client = FakeChatClient(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "retrieve_relevant_content",
                                            "arguments": '{"query":"adult CKD blood pressure target guideline"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
                {"choices": [{"message": {"role": "assistant", "content": "근거 답변 [1]"}}]},
            ]
        )
        retriever = FakeRetriever()
        harness = L2Harness(settings(), chat_client=client, retriever=retriever)

        answer = harness.generate([{"role": "user", "content": "가이드라인 목표는?"}])

        self.assertEqual(answer, "근거 답변 [1]")
        self.assertTrue(
            retriever.queries[0].startswith("adult CKD blood pressure target guideline")
        )
        self.assertIn("가이드라인 목표는?", retriever.queries[0])
        self.assertEqual(client.requests[1]["extra_body"], {})
        grounded_messages = client.requests[1]["messages"]
        self.assertFalse(any(message["role"] == "tool" for message in grounded_messages))
        self.assertIn("status: sufficient", grounded_messages[0]["content"])
        self.assertEqual(grounded_messages[-1]["content"], "가이드라인 목표는?")

    def test_empty_length_response_retries_with_compact_messages(self) -> None:
        client = FakeChatClient(
            [
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": None},
                            "finish_reason": "length",
                        }
                    ]
                },
                {"choices": [{"message": {"role": "assistant", "content": "복구 답변"}}]},
            ]
        )
        harness = L2Harness(settings(), chat_client=client, retriever=FakeRetriever())

        answer = harness.generate([{"role": "user", "content": "감기 관리 방법은?"}])

        self.assertEqual(answer, "복구 답변")
        self.assertEqual(len(client.requests), 2)

    def test_long_drug_label_keeps_query_relevant_excerpt(self) -> None:
        evidence = {
            "cite_uid": "cite-drug",
            "source_type": "dailymed",
            "section": "WARNINGS",
            "content": "unrelated cardiovascular warning. " * 300
            + "Pregnancy warning: avoid ibuprofen because of fetal ductus arteriosus closure.",
        }

        serialized = _serialize_evidence(evidence, "임신 중 이부프로펜 복용", 1_500)

        self.assertLessEqual(len(serialized), 1_500)
        self.assertIn("Pregnancy warning", serialized)
        self.assertIn("ductus arteriosus", serialized)

    def test_nested_guideline_pages_are_included(self) -> None:
        evidence = {
            "cite_uid": "cite-page",
            "source_type": "guideline",
            "pages": [
                {
                    "page": 41,
                    "text": "background material. " * 200
                    + "The recommended blood pressure target is less than 130/80 mm Hg.",
                }
            ],
        }

        serialized = _serialize_evidence(evidence, "blood pressure target", 1_500)

        self.assertIn("pages[0].text", serialized)
        self.assertIn("less than 130/80 mm Hg", serialized)

    def test_retrieval_materializes_selected_citation(self) -> None:
        client = FakeChatClient(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "source-1",
                                        "type": "function",
                                        "function": {
                                            "name": "source_tool",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "final-1",
                                        "type": "function",
                                        "function": {
                                            "name": "finalize_retrieval",
                                            "arguments": (
                                                '{"status":"sufficient","items":['
                                                '{"cite_uid":"cite-1","relevance_score":0.95}],'
                                                '"note":""}'
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            ]
        )
        engine = RetrievalEngine(settings(), chat_client=client, mcp_client=FakeMCPClient())

        result = engine.retrieve("test guideline target")

        self.assertEqual(result.status, "sufficient")
        self.assertEqual(result.mcp_tool_calls, 1)
        self.assertEqual(result.items[0][0].cite_uid, "cite-1")
        self.assertIn("Target statement", result.as_generation_content(10_000))

    def test_relevant_nodes_are_followed_by_bounded_page_fetch(self) -> None:
        client = FakeChatClient(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "node-1",
                                        "type": "function",
                                        "function": {
                                            "name": "index_get_relevant_nodes",
                                            "arguments": (
                                                '{"corpus_tag":"guideline","query":"CKD BP"}'
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        )
        mcp = FakeIndexMCPClient()
        engine = RetrievalEngine(settings(), chat_client=client, mcp_client=mcp)

        result = engine.retrieve("clinical guideline CKD BP target")

        self.assertEqual(result.status, "partial")
        self.assertEqual(result.mcp_tool_calls, 2)
        self.assertEqual(result.items[0][0].cite_uid, "cite-page")
        self.assertEqual(mcp.calls[0][1]["query"], "clinical guideline CKD BP target")
        self.assertEqual(mcp.calls[0][1]["k"], 8)
        self.assertEqual(mcp.calls[1][0], "index_get_page_content")
        self.assertEqual(mcp.calls[1][1]["doc_id"], "doc-goal")
        self.assertEqual(mcp.calls[1][1]["start_page"], 48)
        self.assertEqual(mcp.calls[1][1]["end_page"], 52)


if __name__ == "__main__":
    unittest.main()
