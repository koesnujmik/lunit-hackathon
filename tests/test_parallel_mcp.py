import asyncio
from types import SimpleNamespace
from typing import Any

from l2_baseline.harness import _run_mcp_actions


class ConcurrentMCP:
    def __init__(self) -> None:
        self.active = 0
        self.peak_active = 0
        self.called: list[str] = []

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        self.called.append(name)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return f"{name}:{arguments['query']}"


def _action(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=f'{{"query":"{name}"}}')
    )


def test_mcp_actions_run_concurrently_and_preserve_order() -> None:
    mcp = ConcurrentMCP()
    actions = [_action("first"), _action("second"), _action("third")]

    outputs = asyncio.run(_run_mcp_actions(mcp, actions, remaining_budget=3))  # type: ignore[arg-type]

    assert mcp.peak_active == 3
    assert outputs == ["first:first", "second:second", "third:third"]


def test_mcp_actions_respect_remaining_budget() -> None:
    mcp = ConcurrentMCP()
    actions = [_action("first"), _action("second"), _action("third")]

    outputs = asyncio.run(_run_mcp_actions(mcp, actions, remaining_budget=2))  # type: ignore[arg-type]

    assert mcp.called == ["first", "second"]
    assert len(outputs) == 2
