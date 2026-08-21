from types import SimpleNamespace
from typing import Any

from l2_baseline.workflows import (
    MCPAction,
    RetrievalWorkflowState,
    automatic_next_actions,
    initial_tool_frontier,
    next_tool_frontier,
    prepare_model_actions,
)


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


def _call(name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def test_initial_frontier_replaces_leaf_with_workflow_entry() -> None:
    tools = [
        _tool("index_list_documents", ["corpus_tag", "query"]),
        _tool("index_get_relevant_nodes", ["corpus_tag", "query", "node_id"]),
        _tool(
            "index_get_page_content",
            ["corpus_tag", "doc_id", "start_page", "end_page"],
        ),
        _tool("adr_retrieve_drug_info", ["drug_name"]),
    ]
    candidates = [tools[2], tools[3]]

    frontier = initial_tool_frontier(tools, candidates)

    assert [tool["function"]["name"] for tool in frontier] == [
        "index_list_documents",
        "adr_retrieve_drug_info",
    ]


def test_index_workflow_auto_binds_identifiers_across_steps() -> None:
    tools = [
        _tool("index_list_documents", ["corpus_tag", "query"]),
        _tool("index_get_relevant_nodes", ["corpus_tag", "query", "node_id"]),
        _tool("index_get_document_structure", ["corpus_tag", "node_id"]),
        _tool(
            "index_get_page_content",
            ["corpus_tag", "doc_id", "start_page", "end_page"],
        ),
    ]
    state = RetrievalWorkflowState(query="CKD blood pressure target")
    first = MCPAction(
        tool_name="index_list_documents",
        arguments={"corpus_tag": "guideline", "query": state.query},
    )
    state.observe(first, '{"documents":[{"node_id":"node-1","title":"CKD"}]}')

    second = automatic_next_actions(tools, state)

    assert second == [
        MCPAction(
            tool_name="index_get_relevant_nodes",
            arguments={
                "corpus_tag": "guideline",
                "query": state.query,
                "node_id": "node-1",
            },
        )
    ]

    state.observe(
        second[0],
        '{"nodes":[{"doc_id":"doc-1","start_page":48,"end_page":52}]}',
    )
    third = automatic_next_actions(tools, state)

    assert third == [
        MCPAction(
            tool_name="index_get_page_content",
            arguments={
                "corpus_tag": "guideline",
                "doc_id": "doc-1",
                "start_page": 48,
                "end_page": 52,
            },
        )
    ]


def test_law_workflow_exposes_only_next_dependency() -> None:
    tools = [
        _tool("openapi_law_search", ["query"]),
        _tool("openapi_law_list_articles", ["mst"]),
        _tool("openapi_law_get_article", ["article_key"]),
    ]
    state = RetrievalWorkflowState(query="Medical Service Act informed consent")
    state.observe(
        MCPAction("openapi_law_search", {"query": state.query}),
        '{"results":[{"MST":"law-123"}]}',
    )

    frontier = next_tool_frontier(tools, tools, state)
    actions = automatic_next_actions(tools, state)

    assert [tool["function"]["name"] for tool in frontier] == [
        "openapi_law_list_articles"
    ]
    assert actions == [MCPAction("openapi_law_list_articles", {"mst": "law-123"})]


def test_model_actions_are_bound_and_identical_repeats_are_removed() -> None:
    tools = [_tool("kcd_get_name", ["code", "version"])]
    state = RetrievalWorkflowState(
        query="diabetes code",
        values={"code": "E11", "version": "KCD-9"},
    )
    calls = [_call("kcd_get_name", "{}"), _call("kcd_get_name", "{}")]

    prepared = prepare_model_actions(calls, tools, state)

    assert prepared == [
        MCPAction("kcd_get_name", {"code": "E11", "version": "KCD-9"})
    ]
