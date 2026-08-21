from __future__ import annotations

import argparse
import sys

from .api import APIError
from .answer import generate_answer
from .config import load_settings


def _read_user_message(args_message: list[str]) -> str:
    if args_message:
        return " ".join(args_message).strip()

    stdin_text = sys.stdin.read().strip()
    if not stdin_text:
        raise SystemExit("Pass a message argument or pipe a message on stdin.")
    return stdin_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one answer through the minimal L2 driver.")
    parser.add_argument("message", nargs="*", help="User message to answer.")
    args = parser.parse_args()

    try:
        user_message = _read_user_message(args.message)
        settings = load_settings()
        answer = generate_answer(settings, [{"role": "user", "content": user_message}])
    except (RuntimeError, APIError) as exc:
        raise SystemExit(f"Error: {exc}")

    print(answer)


if __name__ == "__main__":
    main()
