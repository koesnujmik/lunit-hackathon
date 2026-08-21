import argparse
import asyncio

import httpx

from .config import Settings
from .harness import L2Harness


async def next_patient_message(
    client: httpx.AsyncClient, url: str, history: list[dict[str, str]]
) -> str | None:
    """Follow the simulator contract: retry 502 and end a missing conversation on 404."""
    for attempt in range(3):
        response = await client.post(
            url, json={"model": "patient-simulator-ko", "messages": history}
        )
        if response.status_code == 404:
            return None
        if response.status_code != 502:
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        await asyncio.sleep(1 + attempt)
    response.raise_for_status()
    return None


async def simulate(conversations: int, turns: int) -> None:
    settings = Settings()
    harness = L2Harness(settings)
    headers = {"Authorization": f"Bearer {settings.token}"}
    url = settings.patient_api_url.rstrip("/") + "/v1/chat/completions"
    async with httpx.AsyncClient(headers=headers, timeout=settings.request_timeout_sec) as client:
        for case in range(1, conversations + 1):
            history: list[dict[str, str]] = []
            print(f"\n=== conversation {case} ===")
            for _ in range(turns):
                question = await next_patient_message(client, url, history)
                if question is None:
                    break
                history.append({"role": "user", "content": question})
                print(f"환자> {question}")
                answer = await harness.chat(history)
                history.append({"role": "assistant", "content": answer})
                print(f"L2> {answer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Patient Simulator conversations")
    parser.add_argument("--conversations", type=int, default=1)
    parser.add_argument("--turns", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(simulate(args.conversations, min(args.turns, 3)))


if __name__ == "__main__":
    main()
