import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from l2_baseline.config import Settings
from l2_baseline.harness import L2Harness, _response_language_instruction
from l2_baseline.models import RetrievalResult
from l2_baseline.prompts import RETRIEVAL_RATIONALE_SYSTEM_PROMPT


def _harness(responses: list[object]) -> tuple[L2Harness, AsyncMock]:
    create = AsyncMock(side_effect=responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    settings = Settings(LUNIT_FM_API_KEY="lunit_test")
    return L2Harness(settings=settings, client=client), create  # type: ignore[arg-type]


def test_memory_answer_reuses_first_generation_response() -> None:
    message = SimpleNamespace(content="direct answer", tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    harness, create = _harness([response])

    answer = asyncio.run(harness.chat([{"role": "user", "content": "general question"}]))

    assert answer == "direct answer"
    assert create.await_count == 1


def test_response_language_is_explicit() -> None:
    assert "English only" in _response_language_instruction("My knee clicks")
    assert "Korean" in _response_language_instruction("무릎에서 소리가 나요")


def test_retrieval_uses_native_generation_tool_trajectory() -> None:
    call = Mock()
    call.id = "call-1"
    call.function.name = "retrieve_relevant_content"
    call.function.arguments = '{"query":"self-contained query"}'
    call.model_dump.return_value = {
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "retrieve_relevant_content",
            "arguments": call.function.arguments,
        },
    }
    first = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[call]))]
    )
    second = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="grounded answer", tool_calls=None))]
    )
    harness, create = _harness([first, second])
    harness.retrieve = AsyncMock(return_value=RetrievalResult(status="no_evidence"))

    answer = asyncio.run(harness.chat([{"role": "user", "content": "guideline question"}]))

    assert answer == "grounded answer"
    harness.retrieve.assert_awaited_once_with("self-contained query")
    assert create.await_count == 2


def test_tool_selection_allows_model_to_choose_whether_to_call() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[]))]
    )
    harness, create = _harness([response])
    tools = [
        {
            "type": "function",
            "function": {
                "name": "example_tool",
                "description": "Example",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    actions = asyncio.run(harness._choose_actions(tools, "query", "passage"))

    assert actions == []
    assert create.await_args.kwargs["tool_choice"] == "auto"


def test_retrieval_rationale_describes_needed_evidence_without_answering() -> None:
    message = SimpleNamespace(
        content="Need the population, outcome, exact threshold, and official guideline source.",
        tool_calls=None,
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    harness, create = _harness([response])

    rationale = asyncio.run(harness.create_retrieval_rationale("What is the target?"))

    assert "official guideline source" in rationale
    system_prompt = create.await_args.kwargs["messages"][0]["content"]
    assert system_prompt == RETRIEVAL_RATIONALE_SYSTEM_PROMPT
    assert "Do not answer the query" in system_prompt
    assert "under 140 words" in system_prompt


def test_structured_decision_uses_auto_tool_choice() -> None:
    call = Mock()
    call.function.arguments = '{"sufficient":true}'
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[call]))]
    )
    harness, create = _harness([response])
    tool = {
        "type": "function",
        "function": {
            "name": "submit_decision",
            "description": "Submit a decision",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    arguments = asyncio.run(harness._forced_decision("system", "content", tool))

    assert arguments == {"sufficient": True}
    assert create.await_args.kwargs["tool_choice"] == "auto"
