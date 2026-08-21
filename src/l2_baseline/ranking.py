import json
import math
import re
from collections import Counter
from typing import Any

from .models import Evidence

TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣_]+")
CITE_PATTERN = re.compile(r"cite[_-]uid[\"\s:=]+[\"']?([A-Za-z0-9_-]+)", re.IGNORECASE)
CONTEXT_KEYS = {
    "source",
    "source_type",
    "title",
    "document_title",
    "url",
    "doc_id",
    "node_id",
    "page",
    "start_page",
    "end_page",
}


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text.replace("_", " ")) if len(token) > 1]


def _mapping_cite_uid(value: dict[str, Any]) -> str | None:
    for key, item in value.items():
        if key.lower().replace("-", "_") == "cite_uid" and isinstance(item, str):
            return item
    return None


def _json_citable_items(
    value: Any, inherited_context: dict[str, Any] | None = None
) -> list[tuple[str, str]]:
    context = dict(inherited_context or {})
    if isinstance(value, list):
        items: list[tuple[str, str]] = []
        for child in value:
            items.extend(_json_citable_items(child, context))
        return items
    if not isinstance(value, dict):
        return []

    for key, item in value.items():
        if key.lower() in CONTEXT_KEYS and isinstance(item, (str, int, float, bool)):
            context[key] = item

    cite_uid = _mapping_cite_uid(value)
    if cite_uid:
        payload = {**context, **value}
        return [(cite_uid, json.dumps(payload, ensure_ascii=False, sort_keys=True))]

    items = []
    for child in value.values():
        if isinstance(child, (dict, list)):
            items.extend(_json_citable_items(child, context))
    return items


def _text_citable_items(document: str) -> list[tuple[str, str]]:
    matches = list(CITE_PATTERN.finditer(document))
    if not matches:
        return []
    if len(matches) == 1:
        return [(matches[0].group(1), document)]

    shared_prefix = document[: matches[0].start()].strip()[:2000]
    items: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(document)
        item_content = document[match.start() : end].strip()
        if shared_prefix:
            item_content = f"{shared_prefix}\n{item_content}"
        items.append((match.group(1), item_content))
    return items


def split_citable_documents(documents: list[str]) -> list[tuple[str, str]]:
    """Split MCP outputs into independently rankable cite_uid items.

    JSON results retain useful scalar source metadata inherited from ancestor objects. Text results
    fall back to cite_uid boundaries. Repeated UIDs keep the most complete representation.
    """
    by_uid: dict[str, str] = {}
    for document in documents:
        try:
            parsed = json.loads(document)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        items = _json_citable_items(parsed) if parsed is not None else []
        if not items:
            items = _text_citable_items(document)
        for cite_uid, content in items:
            current = by_uid.get(cite_uid)
            if current is None or len(content) > len(current):
                by_uid[cite_uid] = content
    return list(by_uid.items())


def rank_tool_candidates(
    search_text: str, tools: list[dict[str, Any]], limit: int = 6
) -> list[dict[str, Any]]:
    """Lexically prefilter tool schemas; the L2 selector still makes the action decision."""
    if len(tools) <= limit:
        return tools
    tool_texts = [
        " ".join(
            [
                tool["function"]["name"],
                tool["function"].get("description", ""),
                " ".join(tool["function"].get("parameters", {}).get("properties", {})),
            ]
        )
        for tool in tools
    ]
    tokenized = [_tokens(search_text), *(_tokens(text) for text in tool_texts)]
    frequencies = Counter(token for tokens in tokenized[1:] for token in set(tokens))
    idf = {
        token: math.log((1 + len(tools)) / (1 + frequency)) + 1
        for token, frequency in frequencies.items()
    }
    query_counts = Counter(tokenized[0])
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, (tool, tokens) in enumerate(zip(tools, tokenized[1:], strict=True)):
        counts = Counter(tokens)
        score = sum(
            query_counts[token] * count * idf.get(token, 1) ** 2
            for token, count in counts.items()
            if token in query_counts
        )
        scored.append((score, -index, tool))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:limit]]


def rank_documents(passage: str, documents: list[str], top_k: int = 3) -> list[Evidence]:
    """Rank MCP result documents against the HyDE passage with TF-IDF cosine similarity."""
    citable_documents = split_citable_documents(documents)
    if not citable_documents:
        return []
    tokenized = [
        _tokens(passage),
        *(_tokens(content) for _, content in citable_documents),
    ]
    document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
    total = len(tokenized)
    idf = {
        token: math.log((1 + total) / (1 + frequency)) + 1
        for token, frequency in document_frequency.items()
    }

    def vector(tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        return {token: count * idf[token] for token, count in counts.items()}

    query_vector = vector(tokenized[0])
    query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1
    ranked: list[Evidence] = []
    for (cite_uid, content), tokens in zip(citable_documents, tokenized[1:], strict=True):
        doc_vector = vector(tokens)
        doc_norm = math.sqrt(sum(value * value for value in doc_vector.values())) or 1
        score = sum(query_vector.get(token, 0) * value for token, value in doc_vector.items())
        score /= query_norm * doc_norm
        ranked.append(
            Evidence(
                cite_uid=cite_uid,
                relevance_score=max(0, min(1, score)),
                tfidf_score=score,
                content=content,
            )
        )
    ranked.sort(key=lambda item: item.tfidf_score, reverse=True)
    return ranked[:top_k]
