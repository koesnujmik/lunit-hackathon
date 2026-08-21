import asyncio

from .harness import L2Harness


async def run() -> None:
    harness = L2Harness()
    history: list[dict[str, str]] = []
    print("Lunit L2 의료 챗봇입니다. 종료하려면 /quit를 입력하세요.")
    while True:
        query = (await asyncio.to_thread(input, "\n사용자> ")).strip()
        if query in {"/quit", "/exit"}:
            break
        if not query:
            continue
        history.append({"role": "user", "content": query})
        try:
            answer = await harness.chat(history)
        except (RuntimeError, ValueError) as exc:
            history.pop()
            print(f"오류: {exc}")
            continue
        history.append({"role": "assistant", "content": answer})
        print(f"\nL2> {answer}")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
