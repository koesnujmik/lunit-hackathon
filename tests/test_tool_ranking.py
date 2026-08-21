from typing import Any

from l2_baseline.ranking import rank_tool_candidates


def _tool(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_guideline_query_prefers_index_tools() -> None:
    tools = [
        _tool("openapi_law_search", "Search Korean laws"),
        _tool("kcd_search_codes", "Search disease codes"),
        _tool("index_get_relevant_nodes", "Find relevant clinical guideline sections"),
        _tool("index_get_page_content", "Read clinical guideline page content"),
        _tool("adr_retrieve_drug_info", "Retrieve official drug labels"),
    ]
    selected = rank_tool_candidates(
        "current clinical guideline chronic kidney disease blood pressure", tools, limit=2
    )
    names = {tool["function"]["name"] for tool in selected}
    assert names == {"index_get_relevant_nodes", "index_get_page_content"}


def test_prefilter_respects_limit() -> None:
    tools = [_tool(f"tool_{index}", f"description {index}") for index in range(10)]
    assert len(rank_tool_candidates("description", tools, limit=6)) == 6


def test_tool_hybrid_ranking_uses_query_with_higher_weight() -> None:
    tools = [
        _tool("warfarin_tool", "official warfarin interaction"),
        _tool("aspirin_tool", "aspirin bleeding monitoring"),
    ]

    selected = rank_tool_candidates(
        "warfarin interaction",
        tools,
        limit=1,
        rationale="aspirin bleeding monitoring",
    )

    assert selected[0]["function"]["name"] == "warfarin_tool"
