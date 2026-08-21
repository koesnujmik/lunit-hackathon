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


def test_router_uses_latest_turn_for_unrelated_followup() -> None:
    messages = [
        {
            "role": "user",
            "content": "According to current clinical guidelines, what is the BP target?",
        },
        {"role": "assistant", "content": "The guideline recommends a target."},
        {"role": "user", "content": "What kind of exercise would be practical for me?"},
    ]

    assert not _needs_retrieval(messages)


def test_router_keeps_retrieval_for_source_referential_followup() -> None:
    english_messages = [
        {"role": "user", "content": "What do current clinical guidelines recommend?"},
        {"role": "assistant", "content": "The guideline recommends treatment."},
        {
            "role": "user",
            "content": "Does that recommendation apply to older adults?",
        },
    ]
    korean_messages = [
        {"role": "user", "content": "심평원 급여 기준을 알려줘"},
        {"role": "assistant", "content": "현재 급여 기준은 다음과 같습니다."},
        {"role": "user", "content": "그 기준에 예외도 있어?"},
    ]

    assert _needs_retrieval(english_messages)
    assert _needs_retrieval(korean_messages)


def test_router_does_not_treat_causal_source_as_citation_request() -> None:
    assert not _needs_retrieval(
        [{"role": "user", "content": "What could be the source of this shoulder pain?"}]
    )
    assert not _needs_retrieval(
        [
            {
                "role": "user",
                "content": "According to my doctor, this may be muscular. What do you think?",
            }
        ]
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


def test_structured_source_route_skips_l2_tool_selection() -> None:
    harness, create, mcp = _harness([_response(content="E11 [1]")])

    answer = asyncio.run(
        harness.chat([{"role": "user", "content": "당뇨병의 정확한 KCD 코드는?"}])
    )

    assert answer == "E11 [1]"
    assert create.await_count == 1
    assert "tools" not in create.await_args.kwargs
    assert mcp.calls == [
        (
            "kcd_search_codes",
            {"name": "당뇨병", "lang": "auto", "top_k": 5, "revision": "latest"},
        )
    ]


def test_mfds_detail_route_checks_permission_then_indication() -> None:
    harness, create, mcp = _harness([_response(content="허가 적응증 답변 [1] [2]")])

    answer = asyncio.run(
        harness.chat(
            [{"role": "user", "content": "키트루다 식약처 허가사항과 용법용량을 알려줘"}]
        )
    )

    assert answer == "허가 적응증 답변 [1] [2]"
    assert create.await_count == 1
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
