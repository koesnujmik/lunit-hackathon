import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

from .config import Settings

RETRYABLE_STATUS_CODES = {502, 503, 504}
ASSISTANT_TIMEOUT_SEC = 90.0


def _completion_content(response: httpx.Response) -> str:
    response.raise_for_status()
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Completion endpoint returned an invalid response") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Completion endpoint returned an empty response")
    return content.strip()


async def next_patient_message(
    client: httpx.AsyncClient,
    url: str,
    history: list[dict[str, str]],
    *,
    model: str = "patient-simulator-ko",
) -> str | None:
    """Follow the simulator contract: retry transient errors and stop on 404."""
    response: httpx.Response | None = None
    for attempt in range(3):
        response = await client.post(
            url,
            json={
                "model": model,
                "messages": [dict(message) for message in history],
            },
        )
        if response.status_code == 404:
            return None
        if response.status_code not in RETRYABLE_STATUS_CODES:
            return _completion_content(response)
        await asyncio.sleep(1 + attempt)
    assert response is not None
    response.raise_for_status()
    return None


async def next_assistant_message(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    history: list[dict[str, str]],
) -> str:
    """Call the same OpenAI-compatible HTTP surface used by evaluation."""
    response: httpx.Response | None = None
    for attempt in range(2):
        response = await client.post(
            url,
            json={
                "model": model,
                "messages": [dict(message) for message in history],
                "max_tokens": 6_144,
            },
        )
        if response.status_code not in RETRYABLE_STATUS_CODES:
            return _completion_content(response)
        await asyncio.sleep(1 + attempt)
    assert response is not None
    response.raise_for_status()
    raise RuntimeError("Assistant endpoint remained unavailable")


def _normalized_message(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", " ", text.casefold()).strip()


def _is_duplicate_patient_message(
    question: str, history: list[dict[str, str]]
) -> bool:
    normalized = _normalized_message(question)
    return bool(normalized) and any(
        message.get("role") == "user"
        and _normalized_message(message.get("content", "")) == normalized
        for message in history
    )


async def _simulate_conversation(
    *,
    case: int,
    turns: int,
    patient_client: httpx.AsyncClient,
    patient_url: str,
    patient_model: str,
    assistant_client: httpx.AsyncClient,
    assistant_url: str,
    assistant_model: str,
    duplicate_retries: int,
) -> dict[str, Any]:
    history: list[dict[str, str]] = []
    turn_results: list[dict[str, Any]] = []
    total_assistant_sec = 0.0
    print(f"\n=== conversation {case} ===", flush=True)

    for turn_number in range(1, turns + 1):
        patient_started = time.monotonic()
        source = "simulated"
        question = await next_patient_message(
            patient_client,
            patient_url,
            history,
            model=patient_model,
        )
        patient_elapsed = time.monotonic() - patient_started
        if question is None:
            print(
                f"환자 simulator가 {turn_number}턴 전에 대화를 종료했습니다.",
                flush=True,
            )
            break

        for retry in range(duplicate_retries):
            if not _is_duplicate_patient_message(question, history):
                break
            print(
                f"경고: 환자 질문이 이전 질문과 같습니다. 재시도 {retry + 1}/"
                f"{duplicate_retries}",
                flush=True,
            )
            question = await next_patient_message(
                patient_client,
                patient_url,
                history,
                model=patient_model,
            )
            patient_elapsed = time.monotonic() - patient_started
            if question is None:
                break
        if question is None:
            break

        duplicate = _is_duplicate_patient_message(question, history)
        history.append({"role": "user", "content": question})
        print(f"\nTURN {turn_number} 환자 ({source})> {question}", flush=True)
        if duplicate:
            print("경고: 이전 환자 질문과 완전히 동일합니다.", flush=True)

        assistant_started = time.monotonic()
        answer = await next_assistant_message(
            assistant_client,
            assistant_url,
            assistant_model,
            history,
        )
        assistant_elapsed = time.monotonic() - assistant_started
        total_assistant_sec += assistant_elapsed
        history.append({"role": "assistant", "content": answer})
        print(f"L2> {answer}", flush=True)
        print(
            f"TURN {turn_number} 응답 시간: {assistant_elapsed:.3f}초 | "
            f"전달 message: {len(history) - 1}개 | 중복: {duplicate}",
            flush=True,
        )
        turn_results.append(
            {
                "turn": turn_number,
                "patient_source": source,
                "patient": question,
                "patient_generation_sec": round(patient_elapsed, 3),
                "patient_duplicate": duplicate,
                "assistant": answer,
                "assistant_response_sec": round(assistant_elapsed, 3),
                "request_message_count": len(history) - 1,
            }
        )

    completed = len(turn_results)
    average = total_assistant_sec / completed if completed else 0.0
    print("\n--- conversation summary ---", flush=True)
    print(f"완료된 턴: {completed}/{turns}", flush=True)
    print(f"전체 응답 시간: {total_assistant_sec:.3f}초", flush=True)
    print(f"평균 응답 시간: {average:.3f}초", flush=True)
    print(
        f"중복 후속 질문: "
        f"{sum(bool(turn['patient_duplicate']) for turn in turn_results)}개",
        flush=True,
    )
    return {
        "case": case,
        "initial_question": turn_results[0]["patient"] if turn_results else None,
        "completed_turns": completed,
        "requested_turns": turns,
        "total_assistant_sec": round(total_assistant_sec, 3),
        "average_assistant_sec": round(average, 3),
        "turns": turn_results,
        "history": history,
    }


async def simulate(
    conversations: int,
    turns: int,
    *,
    assistant_api_base: str = "http://127.0.0.1:8001/v1",
    assistant_model: str = "Lunit/L2-preview",
    assistant_api_key: str = "local-debug",
    patient_model: str = "patient-simulator-ko",
    duplicate_retries: int = 0,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    settings = Settings()
    patient_headers = {"Authorization": f"Bearer {settings.token}"}
    assistant_headers = {"Authorization": f"Bearer {assistant_api_key}"}
    patient_url = settings.patient_api_url.rstrip("/") + "/v1/chat/completions"
    assistant_url = assistant_api_base.rstrip("/") + "/chat/completions"
    results: list[dict[str, Any]] = []

    patient_timeout = httpx.Timeout(settings.request_timeout_sec)
    assistant_timeout = httpx.Timeout(
        max(ASSISTANT_TIMEOUT_SEC, settings.turn_timeout_sec + 15)
    )
    async with (
        httpx.AsyncClient(
            headers=patient_headers, timeout=patient_timeout
        ) as patient_client,
        httpx.AsyncClient(
            headers=assistant_headers, timeout=assistant_timeout
        ) as assistant_client,
    ):
        for case in range(1, conversations + 1):
            result = await _simulate_conversation(
                case=case,
                turns=max(1, min(turns, 3)),
                patient_client=patient_client,
                patient_url=patient_url,
                patient_model=patient_model,
                assistant_client=assistant_client,
                assistant_url=assistant_url,
                assistant_model=assistant_model,
                duplicate_retries=max(0, duplicate_retries),
            )
            results.append(result)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n시뮬레이션 저장: {output_path}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run official Korean three-turn Patient Simulator conversations"
    )
    parser.add_argument("--conversations", type=int, default=1)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument(
        "--assistant-api-base",
        default="http://127.0.0.1:8001/v1",
        help="OpenAI-compatible candidate API base.",
    )
    parser.add_argument("--assistant-model", default="Lunit/L2-preview")
    parser.add_argument("--assistant-api-key", default="local-debug")
    parser.add_argument("--patient-model", default="patient-simulator-ko")
    parser.add_argument(
        "--duplicate-retries",
        type=int,
        default=0,
        help="Exploratory only: regenerate an exactly repeated patient follow-up.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path for the complete transcript and timing data.",
    )
    args = parser.parse_args()
    asyncio.run(
        simulate(
            max(1, args.conversations),
            max(1, min(args.turns, 3)),
            assistant_api_base=args.assistant_api_base,
            assistant_model=args.assistant_model,
            assistant_api_key=args.assistant_api_key,
            patient_model=args.patient_model,
            duplicate_retries=args.duplicate_retries,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
