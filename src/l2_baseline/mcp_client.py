import json
from contextlib import AsyncExitStack
from typing import Any, Self

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class LunitMCPClient:
    def __init__(self, url: str, token: str, timeout_sec: float = 60) -> None:
        self.url = url
        self.headers = {"Authorization": f"Bearer {token}"}
        self.timeout_sec = timeout_sec
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> Self:
        self._stack = AsyncExitStack()
        read, write, _ = await self._stack.enter_async_context(
            streamable_http_client(self.url, headers=self.headers)
        )
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._stack:
            await self._stack.aclose()

    async def openai_tools(self) -> list[dict[str, Any]]:
        assert self._session
        result = await self._session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in result.tools
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        assert self._session
        result = await self._session.call_tool(name, arguments)
        chunks: list[str] = []
        for part in result.content:
            if hasattr(part, "text"):
                chunks.append(part.text)
            elif hasattr(part, "model_dump"):
                chunks.append(json.dumps(part.model_dump(mode="json"), ensure_ascii=False))
        if result.isError:
            return "MCP tool error: " + "\n".join(chunks)
        return "\n".join(chunks)
