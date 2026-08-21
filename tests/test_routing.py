import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from l2_baseline.config import Settings
from l2_baseline.harness import L2Harness, _response_language_instruction
from l2_baseline.models import RetrievalResult


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
