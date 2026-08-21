import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from l2_baseline.config import Settings
from l2_baseline.harness import L2Harness
from l2_baseline.pipelines import (
    PageTarget,
    article_entries,
    english_ingredients,
    kcd_candidates,
    make_action,
    page_targets,
    select_pipeline_type,
    values_for_keys,
)
from l2_baseline.ranking import rank_documents


def _tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


ResponseFactory = Callable[[dict[str, Any]], str]


class FakeMCP:
    def __init__(self, responses: dict[str, str | ResponseFactory]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        response = self.responses[name]
        return response(arguments) if callable(response) else response


def _harness(root_actions: list[Any]) -> L2Harness:
    harness = L2Harness(
        settings=Settings(LUNIT_FM_API_KEY="lunit_test", L2_MAX_RETRIEVAL_CALLS=5),
        client=SimpleNamespace(),  # type: ignore[arg-type]
    )
    harness._choose_pipeline_actions = AsyncMock(  # type: ignore[method-assign]
        return_value=root_actions
    )
    return harness


def _execute(
    harness: L2Harness,
    mcp: FakeMCP,
    tools: list[dict[str, Any]],
    pipeline_type: str,
    query: str,
) -> tuple[list[str], int]:
    return asyncio.run(
        harness._execute_pipeline(  # type: ignore[arg-type]
            mcp,  # type: ignore[arg-type]
            tools,
            pipeline_type,
            query,
            "",
        )
    )


def test_pipeline_parsers_extract_nested_values_and_ignore_errors() -> None:
    texts = [
        'prefix {"results":[{"MST":"law-123"},{"mst":"law-123"}]} suffix',
    ]

    assert values_for_keys(texts, {"mst"}) == ["law-123"]
    assert english_ingredients(
        [
            (
                '{"items":[{"ingredient_eng":"Warfarin Sodium"},'
                '{"ingredient":"와파린나트륨"}]}'
            )
        ]
    ) == ["Warfarin Sodium"]
    assert kcd_candidates(
        [
            (
                '{"candidates":[{"code":"I10"},{"kcd_code":"I11.0"},'
                '{"code":"not-a-code"}]}'
            )
        ]
    ) == [("I10", None), ("I11.0", None)]


def test_deterministic_pipeline_router_covers_specialized_families() -> None:
    cases = {
        "대한고혈압학회 가이드라인의 혈압 목표": "index",
        "국민건강보험법 시행령 조문": "law",
        "FAERS disproportionality for warfarin": "rag_sql",
        "Search PubMed for CKD blood pressure evidence": "rag_vector",
        "와파린의 DailyMed official label": "drug_label",
        "이 약의 동일성분 대체약": "drug_substitution",
        "고혈압 KCD 코드로 청구 가능한가": "kcd_billing",
        "Warfarin official label interactions": "direct",
    }

    assert {query: select_pipeline_type(query) for query in cases} == cases


def test_page_target_parser_deduplicates_and_limits_page_window() -> None:
    action = make_action(
        "index_get_relevant_nodes",
        {"corpus_tag": "guideline", "query": "CKD target"},
    )
    output = json.dumps(
        {
            "nodes": [
                {"doc_id": "doc-a", "range": [10, 40]},
                {"doc_id": "doc-a", "start_page": 10, "end_page": 29},
                {"doc_id": "doc-b", "page": 7},
                {"doc_id": "invalid", "range": [0, 2]},
            ]
        }
    )

    assert page_targets([(action, output)]) == [
        PageTarget(
            corpus_tag="guideline",
            doc_id="doc-a",
            start_page=10,
            end_page=29,
        ),
        PageTarget(
            corpus_tag="guideline",
            doc_id="doc-b",
            start_page=7,
            end_page=7,
        ),
    ]


def test_article_parser_extracts_unique_keys_with_search_text() -> None:
    entries = article_entries(
        [
            (
                '{"articles":[{"article_key":"art-24","title":"Informed consent"},'
                '{"article_key":"art-99","title":"Penalties"},'
                '{"article_key":"art-24","title":"Duplicate"}]}'
            )
        ]
    )

    assert [key for key, _ in entries] == ["art-24", "art-99"]
    assert "Informed consent" in entries[0][1]


def test_index_pipeline_runs_bounded_root_and_page_calls() -> None:
    query = "CKD blood pressure target"
    root_actions = [
        make_action(
            "index_get_relevant_nodes",
            {"corpus_tag": "guideline", "query": query},
        ),
        make_action(
            "index_keyword_search",
            {"corpus_tag": "guideline", "query": "blood pressure target"},
        ),
    ]

    def page_result(arguments: dict[str, Any]) -> str:
        doc_id = arguments["doc_id"]
        return json.dumps(
            {
                "cite_uid": f"cite-index-{doc_id}",
                "doc_id": doc_id,
                "content": f"{query} evidence from {doc_id}",
            }
        )

    mcp = FakeMCP(
        {
            "index_get_relevant_nodes": json.dumps(
                {
                    "nodes": [
                        {"doc_id": "doc-a", "range": [10, 15]},
                        {"doc_id": "doc-b", "start_page": 20, "end_page": 45},
                    ]
                }
            ),
            "index_keyword_search": json.dumps(
                {
                    "matches": [
                        {"doc_id": "doc-c", "page": 7},
                        {"doc_id": "doc-d", "page": 8},
                    ]
                }
            ),
            "index_get_page_content": page_result,
        }
    )
    harness = _harness(root_actions)
    tools = [
        _tool("index_get_relevant_nodes"),
        _tool("index_keyword_search"),
        _tool("index_get_page_content"),
    ]

    documents, count = _execute(harness, mcp, tools, "index", query)

    assert mcp.calls == [
        (
            "index_get_relevant_nodes",
            {"corpus_tag": "guideline", "query": query},
        ),
        (
            "index_keyword_search",
            {"corpus_tag": "guideline", "query": "blood pressure target"},
        ),
        (
            "index_get_page_content",
            {
                "corpus_tag": "guideline",
                "doc_id": "doc-a",
                "start_page": 10,
                "end_page": 15,
            },
        ),
        (
            "index_get_page_content",
            {
                "corpus_tag": "guideline",
                "doc_id": "doc-b",
                "start_page": 20,
                "end_page": 39,
            },
        ),
        (
            "index_get_page_content",
            {
                "corpus_tag": "guideline",
                "doc_id": "doc-c",
                "start_page": 7,
                "end_page": 7,
            },
        ),
    ]
    assert count == len(mcp.calls) == 5
    assert "cite-index-doc-a" in {item.cite_uid for item in rank_documents(query, "", documents)}


def test_law_pipeline_propagates_mst_and_ranked_article_keys() -> None:
    query = "Medical Service Act informed consent"
    root_action = make_action("openapi_law_search", {"query": query})
    mcp = FakeMCP(
        {
            "openapi_law_search": '{"results":[{"mst":"law-123"}]}',
            "openapi_law_list_articles": (
                '{"articles":[{"article_key":"art-24","title":"Informed consent"},'
                '{"article_key":"art-99","title":"Penalties"}]}'
            ),
            "openapi_law_get_article": (
                '{"cite_uid":"cite-law","content":"Medical Service Act informed consent"}'
            ),
        }
    )
    harness = _harness([root_action])
    tools = [
        _tool("openapi_law_search"),
        _tool("openapi_law_list_articles"),
        _tool("openapi_law_get_article"),
    ]

    documents, count = _execute(harness, mcp, tools, "law", query)

    assert mcp.calls == [
        ("openapi_law_search", {"query": query}),
        ("openapi_law_list_articles", {"mst": "law-123"}),
        (
            "openapi_law_get_article",
            {"mst": "law-123", "article_keys": ["art-24", "art-99"]},
        ),
    ]
    assert count == 3 <= harness.settings.max_retrieval_calls
    assert [item.cite_uid for item in rank_documents(query, "", documents)] == ["cite-law"]


def test_drug_label_pipeline_translates_english_ingredient_for_dailymed() -> None:
    query = "Official interactions for Coumadin"
    root_action = make_action(
        "openapi_mfds_get_drug_indication",
        {"drug_name": "쿠마딘정"},
    )
    mcp = FakeMCP(
        {
            "openapi_mfds_get_drug_indication": (
                '{"items":[{"product_name":"쿠마딘정",'
                '"ingredient_eng":"Warfarin Sodium"}]}'
            ),
            "adr_retrieve_drug_info": (
                '{"cite_uid":"cite-dailymed","title":"Warfarin Sodium label",'
                '"content":"Official interactions for Coumadin"}'
            ),
        }
    )
    harness = _harness([root_action])
    tools = [
        _tool("openapi_mfds_get_drug_indication"),
        _tool("openapi_mfds_check_drug_permission"),
        _tool("adr_retrieve_drug_info"),
    ]

    documents, count = _execute(harness, mcp, tools, "drug_label", query)

    assert mcp.calls == [
        ("openapi_mfds_get_drug_indication", {"drug_name": "쿠마딘정"}),
        ("adr_retrieve_drug_info", {"drug_name": "Warfarin Sodium"}),
    ]
    assert count == 2 <= harness.settings.max_retrieval_calls
    assert [item.cite_uid for item in rank_documents(query, "", documents)] == [
        "cite-dailymed"
    ]


def test_kcd_billing_pipeline_validates_two_codes_within_global_budget() -> None:
    query = "Essential hypertension KCD billing codes"
    root_action = make_action(
        "kcd_search_codes",
        {"query": "essential hypertension", "version": "KCD-9"},
    )

    def terminal_result(arguments: dict[str, Any]) -> str:
        code = arguments["code"]
        cite_code = code.replace(".", "-")
        return json.dumps(
            {
                "cite_uid": f"cite-kcd-{cite_code}",
                "code": code,
                "content": f"{query}: {code}",
            }
        )

    mcp = FakeMCP(
        {
            "kcd_search_codes": (
                '{"candidates":[{"code":"I10"},{"kcd_code":"I11.0"},'
                '{"code":"I12"}]}'
            ),
            "kcd_get_name": terminal_result,
            "openapi_hira_disease_check_code": terminal_result,
        }
    )
    harness = _harness([root_action])
    tools = [
        _tool("kcd_search_codes"),
        _tool("kcd_get_name"),
        _tool("openapi_hira_disease_check_code"),
    ]

    documents, count = _execute(harness, mcp, tools, "kcd_billing", query)

    assert mcp.calls == [
        (
            "kcd_search_codes",
            {"query": "essential hypertension", "version": "KCD-9"},
        ),
        ("kcd_get_name", {"code": "I10"}),
        ("openapi_hira_disease_check_code", {"code": "I10"}),
        ("kcd_get_name", {"code": "I11.0"}),
        ("openapi_hira_disease_check_code", {"code": "I11.0"}),
    ]
    assert count == len(mcp.calls) == harness.settings.max_retrieval_calls == 5
    evidence = rank_documents(query, "", documents)
    assert evidence
    assert all(item.cite_uid.startswith("cite-kcd-") for item in evidence)


def test_drug_substitution_pipeline_checks_candidates_within_budget() -> None:
    query = "same ingredient alternatives for original product"
    root_action = make_action(
        "openapi_mfds_get_drug_indication", {"drug_name": "original"}
    )

    def indication_result(arguments: dict[str, Any]) -> str:
        product = arguments["drug_name"]
        if product == "original":
            return '{"items":[{"ingredient_eng":"medroxyprogesterone"}]}'
        return json.dumps(
            {
                "cite_uid": f"cite-{product}",
                "product_name": product,
                "content": f"approved indication for {product}",
            }
        )

    mcp = FakeMCP(
        {
            "openapi_mfds_get_drug_indication": indication_result,
            "openapi_mfds_find_drugs_by_ingredient": (
                '{"items":[{"product_name":"alternative-a"},'
                '{"product_name":"alternative-b"},'
                '{"product_name":"alternative-c"}]}'
            ),
        }
    )
    harness = _harness([root_action])
    tools = [
        _tool("openapi_mfds_get_drug_indication"),
        _tool("openapi_mfds_check_drug_permission"),
        _tool("openapi_mfds_find_drugs_by_ingredient"),
    ]

    documents, count = _execute(
        harness, mcp, tools, "drug_substitution", query
    )

    assert mcp.calls == [
        ("openapi_mfds_get_drug_indication", {"drug_name": "original"}),
        (
            "openapi_mfds_find_drugs_by_ingredient",
            {"ingredient": "medroxyprogesterone"},
        ),
        (
            "openapi_mfds_get_drug_indication",
            {"drug_name": "alternative-a", "num_rows": 1},
        ),
        (
            "openapi_mfds_get_drug_indication",
            {"drug_name": "alternative-b", "num_rows": 1},
        ),
        (
            "openapi_mfds_get_drug_indication",
            {"drug_name": "alternative-c", "num_rows": 1},
        ),
    ]
    assert count == harness.settings.max_retrieval_calls == 5
    assert rank_documents(query, "", documents)


def test_pipeline_stops_when_required_identifier_is_missing() -> None:
    root_action = make_action(
        "openapi_mfds_get_drug_indication", {"drug_name": "unknown"}
    )
    mcp = FakeMCP(
        {"openapi_mfds_get_drug_indication": '{"items":[],"message":"no match"}'}
    )
    harness = _harness([root_action])
    tools = [
        _tool("openapi_mfds_get_drug_indication"),
        _tool("adr_retrieve_drug_info"),
    ]

    documents, count = _execute(
        harness, mcp, tools, "drug_label", "unknown Korean product"
    )

    assert count == 1
    assert len(documents) == 1
    assert mcp.calls == [
        ("openapi_mfds_get_drug_indication", {"drug_name": "unknown"})
    ]


def test_rag_sql_pipeline_uses_schema_before_generated_query() -> None:
    query = "FAERS reports for warfarin bleeding"
    detail_action = make_action(
        "rag_get_data_source_detail", {"source_name": "faers_12q4_25q4"}
    )
    sql_action = make_action(
        "rag_sql_query",
        {
            "db_name": "faers_12q4_25q4",
            "sql": "SELECT drug_name, event FROM reports LIMIT 5",
        },
    )
    harness = _harness([])
    harness._choose_pipeline_actions = AsyncMock(  # type: ignore[method-assign]
        side_effect=[[detail_action], [sql_action]]
    )
    mcp = FakeMCP(
        {
            "rag_get_data_source_detail": (
                '{"tables":[{"name":"reports","columns":["drug_name","event"]}]}'
            ),
            "rag_sql_query": (
                '{"cite_uid":"cite-faers","content":"warfarin bleeding reports"}'
            ),
        }
    )
    tools = [_tool("rag_get_data_source_detail"), _tool("rag_sql_query")]

    documents, count = _execute(harness, mcp, tools, "rag_sql", query)

    assert mcp.calls == [
        (
            "rag_get_data_source_detail",
            {"source_name": "faers_12q4_25q4"},
        ),
        (
            "rag_sql_query",
            {
                "db_name": "faers_12q4_25q4",
                "sql": "SELECT drug_name, event FROM reports LIMIT 5",
            },
        ),
    ]
    assert count == 2
    assert [item.cite_uid for item in rank_documents(query, "", documents)] == [
        "cite-faers"
    ]


def test_rag_vector_pipeline_uses_collection_metadata_before_search() -> None:
    query = "recent evidence about CKD blood pressure"
    detail_action = make_action(
        "rag_get_data_source_detail", {"source_name": "pubmed_abstracts"}
    )
    vector_action = make_action(
        "rag_vector_query",
        {"collection_name": "pubmed_abstracts", "query": query, "top_k": 5},
    )
    harness = _harness([])
    harness._choose_pipeline_actions = AsyncMock(  # type: ignore[method-assign]
        side_effect=[[detail_action], [vector_action]]
    )
    mcp = FakeMCP(
        {
            "rag_get_data_source_detail": '{"payload_fields":["year","content"]}',
            "rag_vector_query": (
                '{"cite_uid":"cite-pubmed","content":"CKD blood pressure evidence"}'
            ),
        }
    )
    tools = [_tool("rag_get_data_source_detail"), _tool("rag_vector_query")]

    documents, count = _execute(harness, mcp, tools, "rag_vector", query)

    assert mcp.calls == [
        ("rag_get_data_source_detail", {"source_name": "pubmed_abstracts"}),
        (
            "rag_vector_query",
            {"collection_name": "pubmed_abstracts", "query": query, "top_k": 5},
        ),
    ]
    assert count == 2
    assert [item.cite_uid for item in rank_documents(query, "", documents)] == [
        "cite-pubmed"
    ]
