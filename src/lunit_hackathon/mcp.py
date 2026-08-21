from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class MCPError(RuntimeError):
    """Raised when the Lunit MCP endpoint cannot complete a request."""


@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": _truncate(self.description, 900),
                "parameters": _compact_schema(self.input_schema),
            },
        }


_TOOL_CACHE: dict[str, tuple[MCPTool, ...]] = {}
_TOOL_CACHE_LOCK = threading.Lock()


class MCPClient:
    def __init__(self, url: str, api_key: str, timeout_sec: float = 60.0) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self._request_id = 0

    def list_tools(self) -> list[MCPTool]:
        with _TOOL_CACHE_LOCK:
            cached = _TOOL_CACHE.get(self.url)
        if cached is not None:
            return list(cached)

        payload = self._post_rpc("tools/list", {})
        raw_tools = payload.get("tools")
        if not isinstance(raw_tools, list):
            raise MCPError(f"MCP tools/list returned an invalid payload: {payload!r}")

        tools: list[MCPTool] = []
        for raw_tool in raw_tools:
            if not isinstance(raw_tool, dict):
                continue
            name = raw_tool.get("name")
            schema = raw_tool.get("inputSchema")
            if not isinstance(name, str) or not isinstance(schema, dict):
                continue
            description = raw_tool.get("description")
            tools.append(
                MCPTool(
                    name=name,
                    description=description if isinstance(description, str) else "",
                    input_schema=schema,
                )
            )

        if not tools:
            raise MCPError("MCP tools/list returned no usable tools.")
        with _TOOL_CACHE_LOCK:
            _TOOL_CACHE[self.url] = tuple(tools)
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._post_rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError") is True:
            raise MCPError(f"MCP tool {name} reported an error: {tool_result_text(result, 4_000)}")
        return result

    def _post_rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        body = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        last_error: Exception | None = None
        for attempt in range(3):
            request = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                    raw = response.read().decode("utf-8")
                    return self._parse_rpc_response(raw, response.headers.get("Content-Type", ""))
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < 2:
                    last_error = exc
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise MCPError(f"MCP {method} failed with HTTP {exc.code}: {error_body}") from exc
            except TimeoutError as exc:
                raise MCPError(
                    f"MCP {method} timed out after {self.timeout_sec:g}s."
                ) from exc
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise MCPError(f"MCP {method} failed: {exc}") from exc

        raise MCPError(f"MCP {method} failed after retries: {last_error}")

    @staticmethod
    def _parse_rpc_response(raw: str, content_type: str) -> dict[str, Any]:
        try:
            if "text/event-stream" in content_type:
                messages = [
                    json.loads(line[6:])
                    for line in raw.splitlines()
                    if line.startswith("data: ")
                ]
                if not messages:
                    raise ValueError("empty event stream")
                envelope = messages[-1]
            else:
                envelope = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise MCPError("MCP endpoint returned an invalid JSON-RPC response.") from exc

        if not isinstance(envelope, dict):
            raise MCPError(f"MCP endpoint returned an invalid envelope: {envelope!r}")
        if "error" in envelope:
            raise MCPError(f"MCP JSON-RPC error: {envelope['error']!r}")
        result = envelope.get("result")
        if not isinstance(result, dict):
            raise MCPError(f"MCP endpoint returned no result object: {envelope!r}")
        return result


def tool_result_text(result: dict[str, Any], max_chars: int) -> str:
    structured = result.get("structuredContent")
    if structured is not None:
        text = json.dumps(structured, ensure_ascii=False, separators=(",", ":"))
    else:
        blocks = result.get("content")
        text_parts: list[str] = []
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    block_text = block.get("text")
                    if isinstance(block_text, str):
                        text_parts.append(block_text)
        text = "\n".join(text_parts) if text_parts else json.dumps(result, ensure_ascii=False)

    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[tool result truncated after {max_chars} characters]"


def citable_items_from_result(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    roots: list[Any] = []
    structured = result.get("structuredContent")
    if structured is not None:
        roots.append(structured)

    blocks = result.get("content")
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            block_text = block.get("text")
            if not isinstance(block_text, str):
                continue
            try:
                roots.append(json.loads(block_text))
            except json.JSONDecodeError:
                continue

    found: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            cite_uid = value.get("cite_uid")
            if isinstance(cite_uid, str) and cite_uid:
                found[cite_uid] = value
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for root in roots:
        visit(root)
    return found


def _compact_schema(value: Any) -> Any:
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, child in value.items():
            if key == "description" and isinstance(child, str):
                compacted[key] = _truncate(child, 320)
            else:
                compacted[key] = _compact_schema(child)
        return compacted
    if isinstance(value, list):
        return [_compact_schema(item) for item in value]
    return value


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
