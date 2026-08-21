import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    stages: tuple[tuple[str, ...], ...]


WORKFLOWS = (
    WorkflowSpec(
        name="indexed_document",
        stages=(
            ("index_list_documents",),
            ("index_get_relevant_nodes", "index_get_document_structure"),
            ("index_get_page_content",),
        ),
    ),
    WorkflowSpec(
        name="korean_law",
        stages=(
            ("openapi_law_search",),
            ("openapi_law_list_articles",),
            ("openapi_law_get_article",),
        ),
    ),
    WorkflowSpec(
        name="kcd_code",
        stages=(("kcd_search_codes",), ("kcd_get_name",)),
    ),
    WorkflowSpec(
        name="mfds_drug",
        stages=(
            (
                "openapi_mfds_check_drug_permission",
                "openapi_mfds_find_drugs_by_ingredient",
            ),
            ("openapi_mfds_get_drug_indication",),
        ),
    ),
)

TOOL_WORKFLOWS = {
    tool_name: workflow
    for workflow in WORKFLOWS
    for stage in workflow.stages
    for tool_name in stage
}

KEY_ALIASES = {
    "corpus_tag": {"corpus_tag", "corpustag"},
    "doc_id": {"doc_id", "document_id", "docid", "documentid"},
    "node_id": {"node_id", "nodeid"},
    "start_page": {"start_page", "startpage", "page_start", "from_page"},
    "end_page": {"end_page", "endpage", "page_end", "to_page"},
    "mst": {"mst", "mst_id", "law_mst", "law_id"},
    "article_key": {"article_key", "articlekey", "article_id", "articleid"},
    "code": {"code", "kcd_code", "disease_code"},
    "version": {"version", "kcd_version"},
    "product_name": {"product_name", "item_name", "drug_name", "product"},
    "item_seq": {"item_seq", "itemseq", "product_id"},
    "ingredient": {"ingredient", "active_ingredient", "inn"},
}

ALIAS_TO_KEY = {
    alias: canonical for canonical, aliases in KEY_ALIASES.items() for alias in aliases
}

TEXT_STATE_PATTERN = re.compile(
    r"(?i)[\"']?(corpus_tag|doc_id|document_id|node_id|start_page|end_page|"
    r"mst|mst_id|law_mst|article_key|article_id|kcd_code|code|version|"
    r"product_name|item_name|drug_name|item_seq|ingredient)[\"']?\s*[:=]\s*"
    r"[\"']?([^\"',}\]\s]+)"
)


def _normalized_key(key: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z_]", "", key).lower()
    return ALIAS_TO_KEY.get(normalized, normalized)


def _state_pairs(value: Any) -> list[tuple[str, Any]]:
    pairs: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            canonical = _normalized_key(str(key))
            if canonical in KEY_ALIASES and isinstance(child, (str, int, float, bool)):
                pairs.append((canonical, child))
            if isinstance(child, (dict, list)):
                pairs.extend(_state_pairs(child))
    elif isinstance(value, list):
        for child in value:
            pairs.extend(_state_pairs(child))
    return pairs


def _tool_map(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {tool["function"]["name"]: tool for tool in tools}


def action_name_arguments(action: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(action, MCPAction):
        return action.tool_name, action.arguments
    name = action.function.name
    raw_arguments = action.function.arguments or "{}"
    arguments = json.loads(raw_arguments)
    if not isinstance(arguments, dict):
        raise TypeError("MCP action arguments must be an object")
    return name, arguments


@dataclass(frozen=True)
class MCPAction:
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class RetrievalWorkflowState:
    query: str
    values: dict[str, Any] = field(default_factory=dict)
    completed_tools: set[str] = field(default_factory=set)
    active_workflows: set[str] = field(default_factory=set)
    seen_actions: set[str] = field(default_factory=set)

    def value_for(self, property_name: str) -> Any | None:
        if property_name == "query":
            return self.query
        return self.values.get(_normalized_key(property_name))

    def observe(self, action: Any, output: str) -> None:
        tool_name, arguments = action_name_arguments(action)
        workflow = TOOL_WORKFLOWS.get(tool_name)
        if workflow:
            self.active_workflows.add(workflow.name)
        self.completed_tools.add(tool_name)
        self.seen_actions.add(_action_signature(tool_name, arguments))

        for key, value in arguments.items():
            canonical = _normalized_key(key)
            if canonical in KEY_ALIASES:
                self.values[canonical] = value

        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            parsed = None
        parsed_pairs = _state_pairs(parsed) if parsed is not None else []
        if not parsed_pairs:
            parsed_pairs = [
                (_normalized_key(match.group(1)), match.group(2))
                for match in TEXT_STATE_PATTERN.finditer(output)
            ]
        for key, value in parsed_pairs:
            self.values.setdefault(key, value)

    def workflow_context(self) -> str:
        values = json.dumps(self.values, ensure_ascii=False, sort_keys=True)
        completed = ", ".join(sorted(self.completed_tools)) or "none"
        active = ", ".join(sorted(self.active_workflows)) or "none"
        return f"active_workflows={active}\ncompleted_tools={completed}\nstate_values={values}"


def _action_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    serialized = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    return f"{tool_name}:{serialized}"


def _workflow_next_stage(
    workflow: WorkflowSpec, state: RetrievalWorkflowState
) -> tuple[str, ...]:
    for stage in workflow.stages:
        if not any(tool_name in state.completed_tools for tool_name in stage):
            return stage
    return ()


def initial_tool_frontier(
    all_tools: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replace dependent candidate tools with their workflow entry tools."""
    available = _tool_map(all_tools)
    names: list[str] = []
    for candidate in candidates:
        candidate_name = candidate["function"]["name"]
        workflow = TOOL_WORKFLOWS.get(candidate_name)
        desired = workflow.stages[0] if workflow else (candidate_name,)
        for tool_name in desired:
            if tool_name in available and tool_name not in names:
                names.append(tool_name)
    return [available[name] for name in names]


def next_tool_frontier(
    all_tools: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    state: RetrievalWorkflowState,
) -> list[dict[str, Any]]:
    available = _tool_map(all_tools)
    names: list[str] = []
    for workflow in WORKFLOWS:
        if workflow.name not in state.active_workflows:
            continue
        for tool_name in _workflow_next_stage(workflow, state):
            if tool_name in available and tool_name not in names:
                names.append(tool_name)
    if names:
        return [available[name] for name in names]
    return initial_tool_frontier(all_tools, candidates)


def _bind_action(
    tool: dict[str, Any],
    arguments: dict[str, Any],
    state: RetrievalWorkflowState,
) -> MCPAction | None:
    function = tool["function"]
    parameters = function.get("parameters", {})
    required = parameters.get("required", [])
    bound = dict(arguments)
    for property_name in required:
        if property_name in bound:
            continue
        value = state.value_for(property_name)
        if value is None:
            return None
        bound[property_name] = value
    return MCPAction(tool_name=function["name"], arguments=bound)


def prepare_model_actions(
    actions: list[Any],
    all_tools: list[dict[str, Any]],
    state: RetrievalWorkflowState,
) -> list[MCPAction]:
    available = _tool_map(all_tools)
    prepared: list[MCPAction] = []
    batch_signatures: set[str] = set()
    for action in actions:
        try:
            tool_name, arguments = action_name_arguments(action)
        except (json.JSONDecodeError, TypeError):
            continue
        tool = available.get(tool_name)
        if tool is None:
            continue
        bound = _bind_action(tool, arguments, state)
        if bound is None:
            continue
        signature = _action_signature(bound.tool_name, bound.arguments)
        if signature in state.seen_actions or signature in batch_signatures:
            continue
        batch_signatures.add(signature)
        prepared.append(bound)
    return prepared


def automatic_next_actions(
    all_tools: list[dict[str, Any]], state: RetrievalWorkflowState
) -> list[MCPAction]:
    """Build executable next workflow steps without another L2 decision."""
    frontier = next_tool_frontier(all_tools, [], state)
    active_frontier = [
        tool
        for tool in frontier
        if TOOL_WORKFLOWS.get(tool["function"]["name"], WorkflowSpec("", ())).name
        in state.active_workflows
    ]
    prepared: list[MCPAction] = []
    covered_workflows: set[str] = set()
    for tool in active_frontier:
        tool_name = tool["function"]["name"]
        workflow = TOOL_WORKFLOWS[tool_name]
        if workflow.name in covered_workflows:
            continue
        action = _bind_action(tool, {}, state)
        if action is None:
            continue
        signature = _action_signature(action.tool_name, action.arguments)
        if signature in state.seen_actions:
            continue
        covered_workflows.add(workflow.name)
        prepared.append(action)
    return prepared


def has_pending_workflow(state: RetrievalWorkflowState) -> bool:
    return any(
        workflow.name in state.active_workflows and _workflow_next_stage(workflow, state)
        for workflow in WORKFLOWS
    )
