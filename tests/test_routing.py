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
    _citation_audit,
    _compact_messages,
    _deterministic_guideline_request,
    _deterministic_structured_request,
    _guideline_focus_queries,
    _has_explicit_retrieval_intent,
    _index_page_argument_candidates,
    _index_page_arguments,
    _index_relevant_nodes_arguments,
    _prepare_primary_arguments,
    _response_language_instruction,
)
from l2_baseline.models import Evidence, RetrievalResult


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


def _finalize_response(
    cite_uid: str = "cite-1",
    status: str = "sufficient",
    note: str = "",
) -> object:
    items = (
        [{"cite_uid": cite_uid, "relevance_score": 0.95}]
        if status != "no_evidence"
        else []
    )
    return _response(
        tool_calls=[
            _call(
                "finalize_retrieval",
                json.dumps({"status": status, "items": items, "note": note}),
                call_id="finalize-1",
            )
        ]
    )


class FakeMCP:
    def __init__(self) -> None:
        self.tools = [
            _tool("index_list_documents", "List indexed clinical documents"),
            _tool("index_get_relevant_nodes", "Find relevant clinical guideline sections"),
            _tool("index_get_page_content", "Open indexed document pages"),
            _tool("openapi_law_search", "Search Korean laws"),
            _tool("kcd_search_codes", "Search KCD disease codes"),
            _tool("kcd_get_name", "Get a KCD disease name"),
            _tool("adr_retrieve_drug_info", "Retrieve DailyMed drug labels"),
            _tool("hira_updates_search", "Search HIRA reimbursement updates"),
            _tool("openapi_mfds_check_drug_permission", "Check MFDS permission"),
            _tool("openapi_mfds_get_drug_indication", "Get MFDS indication"),
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


class DocumentChainMCP(FakeMCP):
    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        if name == "index_list_documents":
            return json.dumps(
                {
                    "results": [
                        {
                            "node_id": "stroke-guide",
                            "title": "2019 AHA/ASA Acute Ischemic Stroke Guideline",
                            "summary": (
                                "Intravenous alteplase treatment windows, eligibility, and "
                                "contraindications for acute ischemic stroke."
                            ),
                            "score": 0.91,
                        },
                        {
                            "node_id": "unrelated-guide",
                            "title": "Unrelated guideline",
                            "summary": "This guideline does not address thrombolysis.",
                            "score": 0.2,
                        },
                    ]
                }
            )
        if name == "index_get_relevant_nodes":
            return json.dumps(
                [
                    {
                        "doc_id": "stroke-guide",
                        "range": [19, 20],
                        "title": "Time Windows",
                        "summary": (
                            "Alteplase within 4.5 hours for eligible patients and the main "
                            "exclusion criteria."
                        ),
                        "score": 0.94,
                    }
                ]
            )
        if name == "index_get_page_content":
            return json.dumps(
                {
                    "cite_uid": "cite-stroke",
                    "title": "2019 AHA/ASA Acute Ischemic Stroke Guideline",
                    "pages": [
                        {
                            "page": 19,
                            "text": "Eligible patients may receive alteplase in the 3–4.5 hour window.",
                        }
                    ],
                }
            )
        return ""


class CompoundGuidelineMCP(FakeMCP):
    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        if name == "index_get_relevant_nodes":
            if "contraindications" in str(arguments.get("query", "")):
                return json.dumps(
                    [
                        {
                            "doc_id": "stroke-safety",
                            "range": [42, 44],
                            "title": "Alteplase contraindications and exclusions",
                            "summary": "Bleeding, blood pressure, INR, platelet, and surgery exclusions.",
                            "score": 0.96,
                        }
                    ]
                )
            return json.dumps(
                [
                    {
                        "doc_id": "stroke-window",
                        "range": [18, 20],
                        "title": "Alteplase treatment window",
                        "summary": "Treatment within 4.5 hours for selected patients.",
                        "score": 0.95,
                    }
                ]
            )
        if name == "index_get_page_content":
            if arguments.get("doc_id") == "stroke-safety":
                return json.dumps(
                    {
                        "cite_uid": "cite-safety",
                        "content": "Main alteplase contraindications and exclusion criteria.",
                    }
                )
            return json.dumps(
                {
                    "cite_uid": "cite-window",
                    "content": "Selected patients may receive alteplase within 4.5 hours.",
                }
            )
        return ""


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


def test_explicit_retrieval_intent_separates_general_and_source_questions() -> None:
    assert not _has_explicit_retrieval_intent(
        "68세이며 와파린을 복용 중입니다. 무릎 통증에 이부프로펜을 먹어도 되나요?"
    )
    assert _has_explicit_retrieval_intent("최신 진료 지침의 권고와 출처를 알려주세요")
    assert _has_explicit_retrieval_intent("What does the authoritative source recommend?")


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


def test_document_list_selects_a_root_for_relevant_node_search() -> None:
    output = json.dumps(
        {
            "results": [
                {
                    "node_id": "unrelated",
                    "title": "General supportive care",
                    "summary": "No thrombolysis recommendations.",
                    "score": 0.7,
                },
                {
                    "node_id": "stroke-guide",
                    "title": "2019 acute ischemic stroke guideline",
                    "summary": "Alteplase 4.5 hour window and contraindications.",
                    "score": 0.8,
                },
            ]
        }
    )

    arguments = _index_relevant_nodes_arguments(
        "alteplase acute ischemic stroke 4.5 hour contraindications",
        {"corpus_tag": "guideline"},
        output,
    )

    assert arguments is not None
    assert arguments["node_id"] == "stroke-guide"
    assert arguments["corpus_tag"] == "guideline"
    assert arguments["k"] == 8


def test_unambiguous_guideline_route_skips_tool_selector() -> None:
    request = _deterministic_guideline_request(
        "According to current clinical guidelines, what is the CKD target?",
        {"index_get_relevant_nodes"},
    )

    assert request is not None
    assert request[0] == "index_get_relevant_nodes"
    assert request[1]["corpus_tag"] == "guideline"
    assert _deterministic_guideline_request(
        "AHA/ASA 2019 guideline alteplase window and contraindications",
        {"index_get_relevant_nodes"},
    ) is not None
    assert _deterministic_guideline_request(
        "심평원 급여 기준 가이드라인", {"index_get_relevant_nodes"}
    ) is None


def test_original_guideline_context_routes_a_rewritten_query_directly() -> None:
    harness, create, mcp = _harness([_finalize_response()])

    result = asyncio.run(
        harness.retrieve(
            [{"role": "user", "content": "alteplase dosing recommendation"}],
            routing_context="What does the AHA/ASA 2019 guideline recommend?",
        )
    )

    assert result.status == "sufficient"
    assert create.await_count == 1
    assert [name for name, _arguments in mcp.calls] == [
        "index_get_relevant_nodes",
        "index_get_page_content",
    ]
    assert "alteplase dosing recommendation" in mcp.calls[0][1]["query"]


def test_compound_guideline_query_is_split_into_timing_and_safety() -> None:
    queries = _guideline_focus_queries(
        "AHA guideline: give the 4.5 hour treatment window and main contraindications"
    )

    assert len(queries) == 2
    assert "treatment timing" in queries[0]
    assert "last-known-well" in queries[0]
    assert "contraindications" in queries[1]
    assert "coagulation" in queries[1]


def test_structured_routes_use_exact_tool_schemas() -> None:
    tools = {
        "kcd_search_codes",
        "kcd_get_name",
        "adr_retrieve_drug_info",
        "hira_updates_search",
        "openapi_mfds_check_drug_permission",
    }

    assert _deterministic_structured_request(
        "당뇨병의 정확한 KCD-9 코드는?", tools
    ) == (
        "kcd_search_codes",
        {"name": "당뇨병", "lang": "auto", "top_k": 5, "revision": "KCD-9"},
    )
    assert _deterministic_structured_request("KCD-9 E11.9의 공식 명칭은?", tools) == (
        "kcd_get_name",
        {"code": "E11.9", "revision": "KCD-9"},
    )
    assert _deterministic_structured_request(
        "What warnings are in the DailyMed label for aspirin?", tools
    ) == ("adr_retrieve_drug_info", {"drug_name": "aspirin"})


def test_structured_routes_reject_ambiguous_or_multi_source_requests() -> None:
    tools = {
        "adr_retrieve_drug_info",
        "hira_updates_search",
        "openapi_mfds_check_drug_permission",
    }

    assert _deterministic_structured_request("이 약의 DailyMed 라벨을 알려줘", tools) is None
    assert (
        _deterministic_structured_request(
            "키트루다의 식약처 허가와 HIRA 급여 기준을 확인해줘", tools
        )
        is None
    )


def test_citation_audit_detects_only_high_confidence_failures() -> None:
    assert _citation_audit("KCD-9 코드는 E11.9입니다 [1].", evidence_count=1) == []
    assert _citation_audit(
        "정확한 식약처 허가 상태는 확인할 수 없습니다.", evidence_count=0
    ) == []

    issues = _citation_audit(
        "식약처 허가 상태는 유효합니다. 권장 용량은 50 mg입니다 [3].",
        evidence_count=2,
    )
    assert "citation_out_of_range" in issues
    assert "uncited_source_claims:1" in issues


def test_hira_and_mfds_routes_require_a_clear_subject() -> None:
    tools = {"hira_updates_search", "openapi_mfds_check_drug_permission"}

    hira = _deterministic_structured_request(
        "심평원에서 키트루다 비소세포폐암 항암제 급여 기준을 알려줘", tools
    )
    assert hira is not None
    assert hira[0] == "hira_updates_search"
    assert hira[1]["current_only"] is True
    assert hira[1]["document_type"] == "cancer_drug_notice"
    unclear_oncology = _deterministic_structured_request(
        "심평원에서 비소세포폐암 급여 기준을 알려줘", tools
    )
    assert unclear_oncology is not None
    assert unclear_oncology[1]["document_type"] == "all"
    assert _deterministic_structured_request("심평원 급여 기준을 알려줘", tools) is None
    assert _deterministic_structured_request("이 약의 식약처 허가사항은?", tools) is None
    assert _deterministic_structured_request("키트루다 식약처 허가사항은?", tools) == (
        "openapi_mfds_check_drug_permission",
        {"drug_name": "키트루다", "num_rows": 5},
    )


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


def test_index_followup_can_open_two_distinct_aspects() -> None:
    nodes = [
        {
            "doc_id": "stroke-guide",
            "range": [18, 20],
            "title": "Alteplase treatment window",
            "summary": "Selected patients may receive alteplase within 4.5 hours.",
            "score": 0.95,
        },
        {
            "doc_id": "stroke-guide",
            "range": [42, 44],
            "title": "Alteplase contraindications and exclusions",
            "summary": "Bleeding, blood pressure, platelet, INR, and surgery exclusions.",
            "score": 0.93,
        },
        {
            "doc_id": "unrelated-guide",
            "range": [3, 4],
            "title": "Aspirin after stroke",
            "summary": "Antiplatelet treatment after ischemic stroke.",
            "score": 0.4,
        },
    ]

    candidates = _index_page_argument_candidates(
        "AHA guideline alteplase 4.5 hour window and main contraindications",
        {"corpus_tag": "guideline"},
        json.dumps(nodes),
        limit=2,
    )

    assert len(candidates) == 2
    assert {item["start_page"] for item in candidates} == {18, 42}
    assert {item["doc_id"] for item in candidates} == {"stroke-guide"}


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


def test_index_followup_combines_adjacent_complementary_sections() -> None:
    nodes = [
        {
            "doc_id": "stroke-guide",
            "range": [19, 20],
            "title": "Time Windows",
            "summary": "Alteplase treatment within 4.5 hours for eligible patients.",
            "score": 0.8,
        },
        {
            "doc_id": "stroke-guide",
            "range": [21, 22],
            "title": "Bleeding Risk",
            "summary": "Contraindications and exclusions for intravenous alteplase.",
            "score": 0.7,
        },
        {
            "doc_id": "stroke-guide",
            "range": [33, 34],
            "title": "Antiplatelet Treatment",
            "summary": "Aspirin after stroke.",
            "score": 0.6,
        },
    ]

    arguments = _index_page_arguments(
        "alteplase 4.5 hour window and main contraindications",
        {"corpus_tag": "guideline"},
        json.dumps(nodes),
    )

    assert arguments == {
        "corpus_tag": "guideline",
        "doc_id": "stroke-guide",
        "start_page": 19,
        "end_page": 22,
    }


def test_direct_route_uses_one_l2_call() -> None:
    harness, create, mcp = _harness([_response(content="direct answer")])

    answer = asyncio.run(
        harness.chat([{"role": "user", "content": "What commonly causes a sore throat?"}])
    )

    assert answer == "direct answer"
    assert create.await_count == 1
    request = create.await_args.kwargs
    assert "tools" not in request
    assert "tool_choice" not in request
    assert mcp.calls == []


def test_general_question_rejects_spurious_retrieval_request() -> None:
    harness, create, mcp = _harness(
        [
            _response(
                tool_calls=[
                    _call(
                        "retrieve_relevant_content",
                        '{"query":"HIRA reimbursement for knee pain"}',
                    )
                ]
            ),
            _response(content="이부프로펜은 피하고 진료를 받으세요."),
        ]
    )

    answer = asyncio.run(
        harness.chat(
            [
                {
                    "role": "user",
                    "content": (
                        "68세이고 와파린을 복용 중입니다. 무릎 통증에 이부프로펜을 "
                        "먹어도 되나요?"
                    ),
                }
            ]
        )
    )

    assert answer == "이부프로펜은 피하고 진료를 받으세요."
    assert create.await_count == 2
    assert all("tools" not in call.kwargs for call in create.await_args_list)
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
    harness, create, mcp = _harness(
        [
            _finalize_response(),
            _response(content="grounded answer [1]"),
        ]
    )

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

    assert answer == "grounded answer [1]"
    assert create.await_count == 2
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
    finalize_request = create.await_args_list[0].kwargs
    assert finalize_request["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in finalize_request["tools"]] == [
        "finalize_retrieval"
    ]
    final_request = create.await_args_list[1].kwargs
    assert "tool_choice" not in final_request
    assert "tools" not in final_request
    assert "Return only the final user-facing medical answer" in final_request[
        "messages"
    ][0]["content"]
    assert final_request["messages"][-2]["role"] == "assistant"
    assert final_request["messages"][-1]["role"] == "tool"
    assert "cite-1" in final_request["messages"][-1]["content"]
    assert final_request["max_tokens"] == 1_536
    assert "general clinical context" in final_request["messages"][0]["content"]
    assert "answer every part" in final_request["messages"][0]["content"]


def test_citation_repair_adopts_only_an_improved_grounded_answer() -> None:
    harness, create, _mcp = _harness(
        [
            _response(content="The guideline recommends a target below 120 mmHg."),
            _response(
                content="The guideline recommends a target below 120 mmHg [1]."
            ),
        ]
    )
    harness.retrieve = AsyncMock(
        return_value=RetrievalResult(
            status="sufficient",
            evidence=[
                Evidence(
                    cite_uid="cite-1",
                    relevance_score=0.95,
                    content="Treat to a systolic blood pressure target below 120 mmHg.",
                )
            ],
            tool_calls=2,
        )
    )

    answer = asyncio.run(
        harness.chat(
            [
                {
                    "role": "user",
                    "content": "What target does the current CKD guideline recommend?",
                }
            ]
        )
    )

    assert answer == "The guideline recommends a target below 120 mmHg [1]."
    assert create.await_count == 2
    repair_request = create.await_args_list[1].kwargs
    assert "tools" not in repair_request
    assert repair_request["max_tokens"] == 512
    assert "CITATION REPAIR MODE" in repair_request["messages"][0]["content"]
    assert "missing_all_citations" in repair_request["messages"][-1]["content"]


def test_citation_repair_keeps_original_when_rewrite_is_not_better() -> None:
    original = "The guideline recommends a target below 120 mmHg."
    harness, create, _mcp = _harness(
        [
            _response(content=original),
            _response(content="The guideline recommends a target below 130 mmHg."),
        ]
    )
    harness.retrieve = AsyncMock(
        return_value=RetrievalResult(
            status="sufficient",
            evidence=[
                Evidence(
                    cite_uid="cite-1",
                    relevance_score=0.95,
                    content="Treat to a systolic blood pressure target below 120 mmHg.",
                )
            ],
        )
    )

    answer = asyncio.run(
        harness.chat(
            [
                {
                    "role": "user",
                    "content": "What target does the current CKD guideline recommend?",
                }
            ]
        )
    )

    assert answer == original
    assert create.await_count == 2


def test_citation_repair_is_skipped_without_citable_evidence() -> None:
    original = "The official source says alteplase is recommended within 4.5 hours."
    harness, create, _mcp = _harness(
        [
            _response(content=original),
        ]
    )
    harness.retrieve = AsyncMock(
        return_value=RetrievalResult(
            status="no_evidence",
            note="Retrieval timed out before citable evidence was collected.",
        )
    )

    answer = asyncio.run(
        harness.chat(
            [
                {
                    "role": "user",
                    "content": "What does the AHA/ASA 2019 guideline recommend?",
                }
            ]
        )
    )

    assert _citation_audit(original, evidence_count=0)
    assert answer == original
    assert create.await_count == 1


def test_citation_repair_preserves_general_safety_context_for_partial_evidence() -> None:
    original = (
        "The retrieved guideline supports the treatment window [1]. General clinical "
        "context: confirm BP is below 185 mmHg, platelets above 100,000, and INR no "
        "higher than 1.7."
    )
    harness, create, _mcp = _harness([_response(content=original)])
    harness.retrieve = AsyncMock(
        return_value=RetrievalResult(
            status="partial",
            evidence=[
                Evidence(
                    cite_uid="cite-1",
                    relevance_score=0.9,
                    content="Selected patients may be treated within 4.5 hours.",
                )
            ],
        )
    )

    answer = asyncio.run(
        harness.chat(
            [
                {
                    "role": "user",
                    "content": "What does the stroke guideline recommend and what is unsafe?",
                }
            ]
        )
    )

    assert _citation_audit(original, evidence_count=1)
    assert answer == original
    assert create.await_count == 1


def test_text_tool_call_passes_a_self_contained_multiturn_query_to_retrieval() -> None:
    text_call = (
        "<tool_call>retrieve_relevant_content"
        "<arg_key>query</arg_key>"
        "<arg_value>current clinical guideline anticoagulation recommendation for a "
        "68-year-old patient with atrial fibrillation taking warfarin</arg_value>"
        "</tool_call>"
    )
    harness, create, mcp = _harness(
        [
            _response(content=text_call),
            _finalize_response(),
            _response(content="grounded follow-up [1]"),
        ]
    )

    answer = asyncio.run(
        harness.chat(
            [
                {
                    "role": "user",
                    "content": "I am 68, have atrial fibrillation, and take warfarin.",
                },
                {"role": "assistant", "content": "Understood."},
                {
                    "role": "user",
                    "content": "What does the authoritative source recommend about it?",
                },
            ]
        )
    )

    assert answer == "grounded follow-up [1]"
    assert create.await_count == 3
    assert len(mcp.calls) == 2
    assert "68-year-old" in mcp.calls[0][1]["query"]
    assert "atrial fibrillation" in mcp.calls[0][1]["query"]
    assert create.await_args_list[2].kwargs["messages"][-2]["tool_calls"][0][
        "id"
    ].startswith("generation-text")


def test_final_generation_retries_instead_of_leaking_a_text_tool_call() -> None:
    leaked_tool_call = (
        "<tool_call>retrieve_relevant_content"
        "<arg_key>query</arg_key>"
        "<arg_value>acute ischemic stroke alteplase contraindications</arg_value>"
        "</tool_call>"
    )
    harness, create, _mcp = _harness(
        [
            _finalize_response(),
            _response(content=leaked_tool_call),
            _response(content="Grounded clinical answer [1]."),
        ]
    )

    answer = asyncio.run(
        harness.chat(
            [
                {
                    "role": "user",
                    "content": (
                        "According to current clinical guidelines, what is the CKD BP target?"
                    ),
                }
            ]
        )
    )

    assert answer == "Grounded clinical answer [1]."
    assert "<tool_call>" not in answer
    assert create.await_count == 3
    retry_request = create.await_args_list[2].kwargs
    assert "tool_choice" not in retry_request
    assert "tools" not in retry_request
    assert "Do not return tool calls" in retry_request["messages"][0]["content"]
    assert retry_request["messages"][-1]["role"] == "user"
    assert "The retrieval stage is complete" in retry_request["messages"][-1][
        "content"
    ]


def test_truncated_final_generation_is_rewritten_concisely() -> None:
    harness, create, _mcp = _harness(
        [
            _response(
                content="The guideline recommends treatment within 4.5 hours, but",
                finish_reason="length",
            ),
            _response(content="Treat selected patients within 4.5 hours [1]."),
        ]
    )
    harness.retrieve = AsyncMock(
        return_value=RetrievalResult(
            status="sufficient",
            evidence=[
                Evidence(
                    cite_uid="cite-1",
                    relevance_score=0.95,
                    content="Selected patients may be treated within 4.5 hours.",
                )
            ],
        )
    )

    answer = asyncio.run(
        harness.chat(
            [
                {
                    "role": "user",
                    "content": "What does the stroke guideline recommend?",
                }
            ]
        )
    )

    assert answer == "Treat selected patients within 4.5 hours [1]."
    assert create.await_count == 2
    assert create.await_args_list[0].kwargs["max_tokens"] == 1_536
    assert create.await_args_list[1].kwargs["max_tokens"] == 768
    assert "previous draft was cut off" in create.await_args_list[1].kwargs["messages"][
        0
    ]["content"]


def test_structured_source_route_skips_l2_tool_selection() -> None:
    harness, create, mcp = _harness(
        [
            _response(
                tool_calls=[
                    _call(
                        "retrieve_relevant_content",
                        '{"query":"당뇨병의 정확한 KCD 코드"}',
                    )
                ]
            ),
            _finalize_response(),
            _response(content="E11 [1]"),
        ]
    )

    answer = asyncio.run(
        harness.chat([{"role": "user", "content": "당뇨병의 정확한 KCD 코드는?"}])
    )

    assert answer == "E11 [1]"
    assert create.await_count == 3
    assert mcp.calls == [
        (
            "kcd_search_codes",
            {"name": "당뇨병", "lang": "auto", "top_k": 5, "revision": "latest"},
        )
    ]


def test_mfds_detail_route_checks_permission_then_indication() -> None:
    harness, create, mcp = _harness(
        [
            _response(
                tool_calls=[
                    _call(
                        "retrieve_relevant_content",
                        '{"query":"키트루다 식약처 허가사항과 용법용량"}',
                    )
                ]
            ),
            _finalize_response(),
            _response(content="허가 적응증 답변 [1]"),
        ]
    )

    answer = asyncio.run(
        harness.chat(
            [{"role": "user", "content": "키트루다 식약처 허가사항과 용법용량을 알려줘"}]
        )
    )

    assert answer == "허가 적응증 답변 [1]"
    assert create.await_count == 3
    assert mcp.calls == [
        (
            "openapi_mfds_check_drug_permission",
            {"drug_name": "키트루다", "num_rows": 5},
        ),
        (
            "openapi_mfds_get_drug_indication",
            {
                "drug_name": "키트루다",
                "num_rows": 3,
                "include_dosage": True,
                "notice_clause": "투여하지 말",
            },
        ),
    ]


def test_retrieval_failure_still_runs_final_l2_generation() -> None:
    harness, create, _mcp = _harness(
        [
            _response(content="safe answer with limitation"),
        ]
    )
    harness.retrieve = AsyncMock(side_effect=TimeoutError)

    answer = asyncio.run(
        harness.chat(
            [{"role": "user", "content": "According to the guideline, what is recommended?"}]
        )
    )

    assert answer == "safe answer with limitation"
    assert create.await_count == 1
    assert (
        "Retrieval was unavailable"
        in create.await_args.kwargs["messages"][-1]["content"]
    )
    assert create.await_args.kwargs["max_tokens"] == 1_024


def test_retrieval_model_sees_all_mcp_tools_plus_finalize() -> None:
    harness, create, mcp = _harness(
        [
            _response(
                tool_calls=[
                    _call(
                        "openapi_law_search",
                        '{"query":"medical evidence statute"}',
                    )
                ]
            ),
            _finalize_response(),
        ]
    )

    result = asyncio.run(
        harness.retrieve(
            [{"role": "user", "content": "Find authoritative evidence for this request."}]
        )
    )

    assert result.status == "sufficient"
    assert [item.cite_uid for item in result.evidence] == ["cite-1"]
    assert mcp.calls == [
        ("openapi_law_search", {"query": "medical evidence statute"})
    ]
    first_tool_names = {
        tool["function"]["name"] for tool in create.await_args_list[0].kwargs["tools"]
    }
    assert first_tool_names == {
        *(tool["function"]["name"] for tool in mcp.tools),
        "finalize_retrieval",
    }
    assert [
        tool["function"]["name"]
        for tool in create.await_args_list[1].kwargs["tools"]
    ] == ["finalize_retrieval"]


def test_document_list_route_opens_relevant_page_before_finalize() -> None:
    mcp = DocumentChainMCP()
    harness, create, _mcp = _harness(
        [
            _response(
                tool_calls=[
                    _call(
                        "index_list_documents",
                        json.dumps(
                            {
                                "corpus_tag": "guideline",
                                "query": (
                                    "intravenous alteplase acute ischemic stroke 4.5 hour "
                                    "window and contraindications AHA ASA 2019 guideline"
                                ),
                            }
                        ),
                    )
                ]
            ),
            _finalize_response(cite_uid="cite-stroke"),
        ],
        mcp=mcp,
    )

    result = asyncio.run(
        harness.retrieve(
            [
                {
                    "role": "user",
                    "content": (
                        "intravenous alteplase acute ischemic stroke time window 4.5 hours "
                        "eligibility criteria contraindications AHA ASA 2019 indexed source"
                    ),
                }
            ]
        )
    )

    assert result.status == "sufficient"
    assert [item.cite_uid for item in result.evidence] == ["cite-stroke"]
    assert [name for name, _arguments in mcp.calls] == [
        "index_list_documents",
        "index_get_relevant_nodes",
        "index_get_page_content",
    ]
    list_arguments = mcp.calls[0][1]
    assert list_arguments["limit"] == 8
    assert list_arguments["offset"] == 0
    node_arguments = mcp.calls[1][1]
    assert node_arguments["node_id"] == "stroke-guide"
    assert node_arguments["k"] == 8
    assert mcp.calls[2][1] == {
        "corpus_tag": "guideline",
        "doc_id": "stroke-guide",
        "start_page": 19,
        "end_page": 20,
    }
    finalize_request = create.await_args_list[1].kwargs
    assert "cite-stroke" in finalize_request["messages"][-1]["content"]


def test_compound_guideline_retrieval_searches_each_aspect_separately() -> None:
    mcp = CompoundGuidelineMCP()
    harness, create, _mcp = _harness([_response(content="")], mcp=mcp)

    result = asyncio.run(
        harness.retrieve(
            [
                {
                    "role": "user",
                    "content": (
                        "According to the AHA guideline, provide the 4.5 hour alteplase "
                        "window and main contraindications."
                    ),
                }
            ]
        )
    )

    assert result.status == "partial"
    assert {item.cite_uid for item in result.evidence} == {
        "cite-window",
        "cite-safety",
    }
    assert [name for name, _arguments in mcp.calls] == [
        "index_get_relevant_nodes",
        "index_get_page_content",
        "index_get_relevant_nodes",
        "index_get_page_content",
    ]
    assert "treatment timing" in mcp.calls[0][1]["query"]
    assert "contraindications" in mcp.calls[2][1]["query"]
    assert create.await_count == 1


def test_retrieval_keeps_collected_evidence_when_finalize_has_no_time() -> None:
    harness, create, mcp = _harness([])
    harness.settings.retrieval_timeout_sec = 1

    result = asyncio.run(
        harness.retrieve(
            [
                {
                    "role": "user",
                    "content": "According to the guideline, what is the CKD target?",
                }
            ]
        )
    )

    assert result.status == "partial"
    assert [item.cite_uid for item in result.evidence] == ["cite-1"]
    assert create.await_count == 0
    assert [name for name, _arguments in mcp.calls] == [
        "index_get_relevant_nodes",
        "index_get_page_content",
    ]


def test_text_finalize_call_selects_only_resolvable_citations() -> None:
    text_finalize = (
        "<tool_call>finalize_retrieval"
        "<arg_key>status</arg_key><arg_value>sufficient</arg_value>"
        "<arg_key>items</arg_key>"
        '<arg_value>[{"cite_uid":"cite-1","relevance_score":0.9},'
        '{"cite_uid":"invented","relevance_score":0.8}]</arg_value>'
        "<arg_key>note</arg_key><arg_value></arg_value>"
        "</tool_call>"
    )
    harness, _create, _mcp = _harness([_response(content=text_finalize)])

    result = asyncio.run(
        harness.retrieve(
            [
                {
                    "role": "user",
                    "content": "According to clinical guidelines, what is the CKD target?",
                }
            ]
        )
    )

    assert result.status == "partial"
    assert [item.cite_uid for item in result.evidence] == ["cite-1"]
    assert "invented" in result.note
