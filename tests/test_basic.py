from __future__ import annotations

import unittest
from typing import Any

from lunit_hackathon.answer import fallback_answer, generate_answer
from lunit_hackathon.api import APIError
from lunit_hackathon.config import Settings


def settings() -> Settings:
    return Settings(
        fm_api_url="https://model.example",
        fm_api_key="test-key",
        fm_model="Lunit/L2-preview",
        patient_api_url="https://patient.example",
        patient_model="patient-simulator-ko",
        timeout_sec=1,
        max_tokens=2_048,
    )


class FakeClient:
    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def chat_completions(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class BasicDriverTests(unittest.TestCase):
    def test_normal_path_is_one_l2_call_without_tools(self) -> None:
        client = FakeClient(
            [{"choices": [{"message": {"role": "assistant", "content": "답변"}}]}]
        )

        answer = generate_answer(
            settings(),
            [{"role": "user", "content": "질문"}],
            client=client,
        )

        self.assertEqual(answer, "답변")
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0]["extra_body"], {})
        self.assertNotIn("tools", client.requests[0])

    def test_api_error_retries_once_with_compact_history(self) -> None:
        client = FakeClient(
            [
                APIError("temporary error"),
                {"choices": [{"message": {"role": "assistant", "content": "복구 답변"}}]},
            ]
        )
        conversation = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * 5_000}
            for index in range(8)
        ]

        answer = generate_answer(settings(), conversation, client=client)

        self.assertEqual(answer, "복구 답변")
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(len(client.requests[1]["messages"]), 5)
        self.assertLessEqual(len(client.requests[1]["messages"][-1]["content"]), 2_500)

    def test_last_resort_fallback_matches_basic_language(self) -> None:
        korean = fallback_answer([{"role": "user", "content": "도와주세요"}])
        english = fallback_answer([{"role": "user", "content": "Please help"}])

        self.assertIn("죄송", korean)
        self.assertIn("sorry", english)


if __name__ == "__main__":
    unittest.main()
