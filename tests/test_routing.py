import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from l2_baseline.config import Settings
from l2_baseline.harness import L2Harness, _response_language_instruction


def _harness(responses: list[object]) -> tuple[L2Harness, AsyncMock]:
    create = AsyncMock(side_effect=responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    settings = Settings(LUNIT_FM_API_KEY="lunit_test")
    return L2Harness(settings=settings, client=client), create  # type: ignore[arg-type]


def test_response_language_is_explicit() -> None:
    assert "English only" in _response_language_instruction("My knee clicks")
    assert "Korean" in _response_language_instruction("무릎에서 소리가 나요")


def test_direct_mode_uses_one_call_with_compact_history() -> None:
    message = SimpleNamespace(content="direct answer", tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    harness, create = _harness([response])
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "more context"},
        {"role": "assistant", "content": "more response"},
        {"role": "user", "content": "latest question"},
    ]

    answer = asyncio.run(harness.chat(messages))

    assert answer == "direct answer"
    assert create.await_count == 1
    request = create.await_args.kwargs
    assert "tools" not in request
    assert request["temperature"] == 0
    assert request["max_tokens"] == 1_536
    assert len(request["messages"]) == 5
    assert request["messages"][1:] == messages[-4:]


def test_direct_mode_truncates_each_message() -> None:
    message = SimpleNamespace(content="answer", tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    harness, create = _harness([response])

    asyncio.run(harness.chat([{"role": "user", "content": "x" * 7_000}]))

    sent = create.await_args.kwargs["messages"][-1]["content"]
    assert len(sent) == 6_000
