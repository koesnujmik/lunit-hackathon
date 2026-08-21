import asyncio
from unittest.mock import AsyncMock, Mock

from l2_baseline.simulator import (
    _is_duplicate_patient_message,
    _simulate_conversation,
    next_patient_message,
)


def test_patient_simulator_retries_502() -> None:
    failed = Mock(status_code=502)
    success = Mock(status_code=200)
    success.json.return_value = {"choices": [{"message": {"content": "질문"}}]}
    client = AsyncMock()
    client.post.side_effect = [failed, success]

    result = asyncio.run(next_patient_message(client, "https://example.test", []))
    assert result == "질문"
    assert client.post.await_count == 2


def test_duplicate_patient_message_normalizes_case_and_punctuation() -> None:
    history = [
        {"role": "user", "content": "Is this urgent?"},
        {"role": "assistant", "content": "Tell me more."},
    ]

    assert _is_duplicate_patient_message("  IS this urgent!! ", history) is True
    assert _is_duplicate_patient_message("My symptoms changed.", history) is False


def test_official_three_turn_simulation_preserves_full_history() -> None:
    patient_client = AsyncMock()
    patient_responses = []
    for question in (
        "Official opening question",
        "Follow-up question two",
        "Follow-up question three",
    ):
        response = Mock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": question}}]
        }
        patient_responses.append(response)
    patient_client.post.side_effect = patient_responses

    assistant_client = AsyncMock()
    assistant_responses = []
    for answer in ("Answer one", "Answer two", "Answer three"):
        response = Mock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": answer}}]
        }
        assistant_responses.append(response)
    assistant_client.post.side_effect = assistant_responses

    result = asyncio.run(
        _simulate_conversation(
            case=1,
            turns=3,
            patient_client=patient_client,
            patient_url="https://patient.test/v1/chat/completions",
            patient_model="patient-simulator-ko",
            assistant_client=assistant_client,
            assistant_url="http://candidate.test/v1/chat/completions",
            assistant_model="Lunit/L2-preview",
            duplicate_retries=0,
        )
    )

    assert result["completed_turns"] == 3
    assert result["initial_question"] == "Official opening question"
    assert [turn["patient_source"] for turn in result["turns"]] == ["simulated"] * 3
    assert [
        len(call.kwargs["json"]["messages"])
        for call in assistant_client.post.await_args_list
    ] == [1, 3, 5]
    assert [
        len(call.kwargs["json"]["messages"])
        for call in patient_client.post.await_args_list
    ] == [0, 2, 4]
    assert result["history"][-1] == {
        "role": "assistant",
        "content": "Answer three",
    }
