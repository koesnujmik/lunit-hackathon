from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .answer import generate_answer
from .api import APIError, OpenAICompatibleClient, first_message_content
from .config import Settings, load_settings


def generate_patient_message(settings: Settings, conversation: list[dict[str, Any]]) -> str:
    client = OpenAICompatibleClient(
        base_url=settings.patient_api_url,
        api_key=settings.fm_api_key,
        timeout_sec=settings.timeout_sec,
    )
    response = client.chat_completions(
        model=settings.patient_model,
        messages=conversation,
        temperature=0.7,
    )
    return first_message_content(response).strip()


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a basic Patient Simulator conversation.")
    parser.add_argument("--turns", type=int, default=3, help="Number of user/assistant turns to run.")
    parser.add_argument("--log-dir", default="logs", help="Directory for JSONL conversation logs.")
    args = parser.parse_args()

    if args.turns < 1:
        raise SystemExit("--turns must be at least 1.")

    try:
        settings = load_settings()
        conversation: list[dict[str, Any]] = []
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = Path(args.log_dir) / f"simulator_{run_id}.jsonl"

        for turn_index in range(1, args.turns + 1):
            patient_message = generate_patient_message(settings, conversation)
            conversation.append({"role": "user", "content": patient_message})
            append_jsonl(log_path, {"turn": turn_index, "role": "user", "content": patient_message})

            assistant_message = generate_answer(settings, conversation)
            conversation.append({"role": "assistant", "content": assistant_message})
            append_jsonl(log_path, {"turn": turn_index, "role": "assistant", "content": assistant_message})

            print(f"\n## Turn {turn_index}")
            print(f"\nUser:\n{patient_message}")
            print(f"\nAssistant:\n{assistant_message}")

        print(f"\nLog: {log_path}")
    except (RuntimeError, APIError) as exc:
        raise SystemExit(f"Error: {exc}")


if __name__ == "__main__":
    main()
