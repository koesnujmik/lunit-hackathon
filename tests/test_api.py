from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from httpx import Request, Response
from openai import InternalServerError

from l2_baseline import api


class FakeHarness:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []
        self.settings = SimpleNamespace(turn_timeout_sec=1)

    async def chat(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return "테스트 답변"


def test_models_endpoint() -> None:
    response = TestClient(api.app).get("/v1/models")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert response.json()["data"][0]["id"] == "Lunit/L2-preview"


def test_health_identifies_contract_aware_retrieval_mode() -> None:
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json()["mode"] == "contract-aware-bounded-retrieval"
    assert response.json()["model_concurrency"] == 8


def test_chat_completions_preserves_full_history(monkeypatch: object) -> None:
    harness = FakeHarness()
    monkeypatch.setattr(api, "_harness", harness)  # type: ignore[attr-defined]
    request = {
        "model": "Lunit/L2-preview",
        "messages": [
            {"role": "user", "content": "첫 질문"},
            {"role": "assistant", "content": "첫 답변"},
            {"role": "user", "content": "후속 질문"},
        ],
    }
    response = TestClient(api.app).post("/v1/chat/completions", json=request)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "테스트 답변",
    }
    assert harness.messages == request["messages"]


def test_streaming_is_openai_compatible(monkeypatch: object) -> None:
    monkeypatch.setattr(api, "_harness", FakeHarness())  # type: ignore[attr-defined]
    response = TestClient(api.app).post(
        "/v1/chat/completions",
        json={
            "model": "Lunit/L2-preview",
            "messages": [{"role": "user", "content": "질문"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"content": "테스트 답변"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_upstream_failure_is_reported_as_bad_gateway(monkeypatch: object) -> None:
    harness = FakeHarness()
    harness.chat = AsyncMock(
        side_effect=InternalServerError(
            "upstream unavailable",
            response=Response(502, request=Request("POST", "https://model.test")),
            body=None,
        )
    )
    monkeypatch.setattr(api, "_harness", harness)  # type: ignore[attr-defined]
    response = TestClient(api.app).post(
        "/v1/chat/completions",
        json={
            "model": "Lunit/L2-preview",
            "messages": [{"role": "user", "content": "question"}],
        },
    )
    assert response.status_code == 502


def test_turn_timeout_is_bounded(monkeypatch: object) -> None:
    harness = FakeHarness()
    harness.chat = AsyncMock(side_effect=TimeoutError())
    monkeypatch.setattr(api, "_harness", harness)  # type: ignore[attr-defined]

    response = TestClient(api.app).post(
        "/v1/chat/completions",
        json={
            "model": "Lunit/L2-preview",
            "messages": [{"role": "user", "content": "question"}],
        },
    )

    assert response.status_code == 504
