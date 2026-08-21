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
    _citable_documents,
    _compact_messages,
    _deterministic_guideline_request,
    _generation_token_budget,
    _guarded_answer_problem,
    _index_page_arguments,
    _listed_document_node_arguments,
    _looks_truncated,
    _needs_retrieval,
    _precision_retrieval_tools,
    _prepare_precision_vector_arguments,
    _prepare_primary_arguments,
    _response_contract_instruction,
    _response_language_instruction,
    _should_verify,
    _truncate_middle,
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
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


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
        if name == "kcd_search_codes":
            return '{"cite_uid":"cite-1","content":"당뇨병 KCD code E11"}'
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


def test_artifact_request_takes_priority_over_incidental_guideline_language() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Write a case-management progress note documenting a parent's request for "
                "a personalized recommended vaccination schedule."
            ),
        }
    ]

    assert not _needs_retrieval(messages)
    assert "ARTIFACT MODE" in _response_contract_instruction(messages)


def test_missing_jurisdiction_is_clarified_before_legal_retrieval() -> None:
    messages = [
        {
            "role": "user",
            "content": "Can a pharmacist prescribe orlistat or antibiotics?",
        }
    ]

    assert not _needs_retrieval(messages)
    assert "JURISDICTION MISSING" in _response_contract_instruction(messages)


def test_jurisdiction_gate_generates_only_a_short_clarification() -> None:
    harness, create, mcp = _harness(
        [
            _response(
                tool_calls=[
                    _call(
                        "request_jurisdiction",
                        '{"question":"Which country and state or province are you in?"}',
                    )
                ]
            )
        ]
    )

    answer = asyncio.run(
        harness.chat([{"role": "user", "content": "Can a pharmacist prescribe antibiotics?"}])
    )

    assert answer == "Which country and state or province are you in?"
    assert create.await_args.kwargs["max_tokens"] == 512
    assert create.await_args.kwargs["tool_choice"] == "required"
    assert mcp.calls == []


def test_jurisdiction_gate_retries_a_search_phrase_that_assumes_a_country() -> None:
    harness, create, _mcp = _harness(
        [
            _response(
                tool_calls=[
                    _call(
                        "request_jurisdiction",
                        '{"question":"대한민국 약사법상 처방 권한"}',
                    )
                ]
            ),
            _response(
                tool_calls=[
                    _call(
                        "request_jurisdiction",
                        '{"question":"어느 국가와 지역에서 처방받으려는 건가요?"}',
                    )
                ]
            ),
        ]
    )

    answer = asyncio.run(
        harness.chat([{"role": "user", "content": "약사가 항생제를 처방할 수 있나요?"}])
    )

    assert answer == "어느 국가와 지역에서 처방받으려는 건가요?"
    assert create.await_count == 2


def test_repeated_single_value_request_is_verified_and_tightly_bounded() -> None:
    messages = [
        {"role": "user", "content": "Give me the daily amount in grams."},
        {"role": "assistant", "content": "It may be 5% or 10%."},
        {
            "role": "user",
            "content": "That is confusing. Give me one final number in grams only, briefly.",
        },
    ]
    harness, create, _mcp = _harness(
        [
            _response(
                tool_calls=[
                    _call(
                        "submit_single_value",
                        '{"value":25,"unit":"g per day"}',
                    )
                ]
            ),
        ]
    )

    answer = asyncio.run(harness.chat(messages))

    assert answer == "25 g per day"
    assert create.await_count == 1
    assert _generation_token_budget(messages, default=1_536, concise=384) == 384
    assert "rejected ranges" in create.await_args_list[0].kwargs["messages"][0]["content"]


def test_health_data_task_contract_preserves_data_and_missing_fields() -> None:
    contract = _response_contract_instruction(
        [
            {
                "role": "user",
                "content": "이 검사 결과를 날짜별 표로 정리하고 BMI를 계산해줘.",
            }
        ]
    )

    assert "HEALTH-DATA MODE" in contract
    assert "preserve dates and units" in contract


def test_resource_limited_contract_blocks_lay_invasive_procedures() -> None:
    contract = _response_contract_instruction(
        [
            {
                "role": "user",
                "content": "외딴섬에서 발 감염이 낫지 않는데 병원이 멉니다.",
            }
        ]
    )

    assert "RESOURCE-LIMITED MODE" in contract
    assert "cut, drain, debride" in contract
    assert _guarded_answer_problem(
        "Use ceftriaxone 2 g IV.", resource_limited=True, missing_data=False
    )


def test_missing_data_risk_assessment_is_verified() -> None:
    assert _guarded_answer_problem(
        "현재 중증 합병증 가능성은 낮습니다.",
        resource_limited=False,
        missing_data=True,
    )


def test_guarded_route_retries_unsafe_output_before_returning() -> None:
    harness, create, _mcp = _harness(
        [
            _response(
                tool_calls=[
                    _call(
                        "submit_safe_answer",
                        '{"final_answer":"Use ceftriaxone 2 g IV."}',
                    )
                ]
            ),
            _response(
                tool_calls=[
                    _call(
                        "submit_safe_answer",
                        '{"final_answer":"상처를 깨끗한 물로 씻고 덮은 뒤 진료 경로를 마련하세요."}',
                    )
                ]
            ),
        ]
    )

    answer = asyncio.run(
        harness.chat([{"role": "user", "content": "외딴섬에서 발 감염이 낫지 않습니다."}])
    )

    assert answer == "상처를 깨끗한 물로 씻고 덮은 뒤 진료 경로를 마련하세요."
    assert create.await_count == 2
    assert create.await_args.kwargs["tool_choice"] == "required"


def test_artifact_answer_uses_contract_without_extra_verifier_call() -> None:
    messages = [
        {
            "role": "user",
            "content": "Write a case-management progress note about the vaccine inquiry.",
        }
    ]
    harness, create, _mcp = _harness(
        [
            _response(
                content="Progress note: Parent called about immunization records. Plan: follow up."
            ),
        ]
    )

    answer = asyncio.run(harness.chat(messages))

    assert answer.startswith("Progress note:")
    assert create.await_count == 1


def test_precise_clinician_dosing_routes_to_bounded_evidence() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "I am a surgeon. What cefazolin dose and intraoperative redosing interval "
                "should I use for a patient with ESRD?"
            ),
        }
    ]

    assert _needs_retrieval(messages)
    assert _should_verify(messages)


def test_precision_retrieval_constrains_selector_to_pubmed() -> None:
    tools = [
        _tool("rag_vector_query", "Search PubMed abstracts"),
        _tool("adr_retrieve_drug_info", "Search label warnings"),
    ]

    selected = _precision_retrieval_tools(
        "What cefazolin dose and redosing interval should be used?", tools
    )

    assert [tool["function"]["name"] for tool in selected] == ["rag_vector_query"]
    assert _precision_retrieval_tools("Check the DailyMed dose", tools) == tools
    assert _prepare_precision_vector_arguments(
        {"collection_name": "dailymed_26_08", "query": "cefazolin", "filters": {}}
    ) == {
        "collection_name": "pubmed_abstracts",
        "query": "cefazolin",
        "top_k": 5,
    }


def test_compaction_preserves_first_user_request_and_recent_turns() -> None:
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"message-{index}"}
        for index in range(11)
    ]

    compact = _compact_messages(messages, max_messages=6, max_chars=4_000)

    assert len(compact) == 6
    assert compact[0] == messages[0]
    assert compact[1:] == messages[-5:]


def test_tool_result_compaction_preserves_trailing_citation_metadata() -> None:
    result = '{"content":"' + ("x" * 8_000) + '","cite_uid":"cite-tail"}'

    compact = _truncate_middle(result, 1_000)

    assert "cite-tail" in compact
    assert len(compact) <= 1_000


def test_vector_results_become_individual_citation_blocks() -> None:
    output = json.dumps(
        {
            "items": [
                {"cite_uid": "cite-a", "title": "First", "content": "one"},
                {"cite_uid": "cite-b", "title": "Second", "content": "two"},
            ],
            "message": "",
        }
    )

    documents = _citable_documents(output, 1_000)

    assert len(documents) == 2
    assert "cite-a" in documents[0]
    assert "cite-b" in documents[1]


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
    assert (
        _deterministic_guideline_request(
            "심평원 급여 기준 가이드라인", {"index_get_relevant_nodes"}
        )
        is None
    )


def test_named_guideline_document_selection_requires_all_authorities() -> None:
    documents = [
        {
            "node_id": "wrong",
            "title": "AHA ACC ADA ASN cardiorenal guideline",
            "summary": "General CKD definitions",
        },
        {
            "node_id": "right",
            "title": "ADA EASD consensus report",
            "summary": "Type 2 diabetes treatment in chronic kidney disease",
        },
    ]

    arguments = _listed_document_node_arguments(
        "ADA/EASD guideline for type 2 diabetes and CKD",
        json.dumps(documents),
    )

    assert arguments is not None
    assert arguments["node_id"] == "right"

    assert (
        _listed_document_node_arguments(
            "ADA/EASD guideline for type 2 diabetes and CKD",
            json.dumps(documents[:1]),
        )
        is None
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
    harness, create, _mcp = _harness([_response(content=""), _response(content="recovered answer")])

    answer = asyncio.run(harness.chat([{"role": "user", "content": "general question"}]))

    assert answer == "recovered answer"
    assert create.await_count == 2
    retry_request = create.await_args_list[1].kwargs
    assert "RECOVERY INSTRUCTION" in retry_request["messages"][0]["content"]
    assert retry_request["max_tokens"] == 512


def test_raw_tool_markup_is_retried_without_losing_answer_contract() -> None:
    harness, create, _mcp = _harness(
        [
            _response(
                content=(
                    "<tool_call>mcp__some_tool<arg_key>query</arg_key>"
                    "<arg_value>x</arg_value></tool_call>"
                )
            ),
            _response(content="A safe, useful answer."),
        ]
    )

    answer = asyncio.run(harness.chat([{"role": "user", "content": "What should I do next?"}]))

    assert answer == "A safe, useful answer."
    retry_prompt = create.await_args_list[1].kwargs["messages"][0]["content"]
    assert "TURN-SPECIFIC RESPONSE CONTRACT" in retry_prompt
    assert "RECOVERY INSTRUCTION" in retry_prompt


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


def test_stop_response_with_dangling_numeric_range_is_retried() -> None:
    harness, create, _mcp = _harness(
        [
            _response(content="Keep indoor humidity around 40~", finish_reason="stop"),
            _response(content="Keep indoor humidity around 40% to 60%.", finish_reason="stop"),
        ]
    )

    answer = asyncio.run(harness.chat([{"role": "user", "content": "Give me brief home care."}]))

    assert answer.endswith("60%.")
    assert create.await_count == 2
    assert create.await_args_list[1].kwargs["max_tokens"] == 1_024
    assert _looks_truncated("Keep indoor humidity around 40~", "stop")
    assert _looks_truncated("증상이 심하거나", "stop") is False
    assert _looks_truncated(("기본 관리를 설명했습니다. " * 8) + "증상이 심하거나", "stop")
    assert not _looks_truncated(("기본 관리를 설명했습니다. " * 8) + "진료를 받으세요", "stop")


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

    answer = asyncio.run(harness.chat([{"role": "user", "content": "당뇨병의 정확한 KCD 코드는?"}]))

    assert answer == "E11 [1]"
    assert create.await_count == 2
    assert create.await_args_list[0].kwargs["tool_choice"] == "required"
    assert mcp.calls == [("kcd_search_codes", {"query": "당뇨병"})]


def test_tool_selector_receives_complete_mcp_tool_set() -> None:
    mcp = FakeMCP()
    mcp.tools.extend(
        _tool(f"extra_tool_{index}", f"Specialized source {index}") for index in range(12)
    )
    harness, create, _mcp = _harness(
        [
            _response(tool_calls=[_call("kcd_search_codes", '{"query":"당뇨병"}')]),
            _response(content="E11 [1]"),
        ],
        mcp=mcp,
    )

    answer = asyncio.run(harness.chat([{"role": "user", "content": "당뇨병의 정확한 KCD 코드는?"}]))

    assert answer == "E11 [1]"
    assert len(create.await_args_list[0].kwargs["tools"]) == len(mcp.tools)


def test_verifier_returns_checked_user_facing_answer() -> None:
    verified = {"final_answer": "Use the evidence-supported interval."}
    harness, create, _mcp = _harness(
        [_response(tool_calls=[_call("submit_verified_answer", json.dumps(verified))])]
    )

    answer = asyncio.run(
        harness._verify_answer(
            [{"role": "user", "content": "I am a surgeon. Check this dose."}],
            "Unverified draft",
        )
    )

    assert answer == "Use the evidence-supported interval."
    assert create.await_args.kwargs["tool_choice"] == "required"


def test_primary_timeout_uses_l2_deadline_fallback() -> None:
    harness, create, _mcp = _harness([_response(content="useful fallback answer")])
    harness._chat_primary = AsyncMock(side_effect=TimeoutError)  # type: ignore[method-assign]

    answer = asyncio.run(harness.chat([{"role": "user", "content": "What should I do next?"}]))

    assert answer == "useful fallback answer"
    assert "DEADLINE FALLBACK" in create.await_args.kwargs["messages"][0]["content"]


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
    assert (
        "Never return only a retrieval-failure disclaimer"
        in (create.await_args.kwargs["messages"][0]["content"])
    )
