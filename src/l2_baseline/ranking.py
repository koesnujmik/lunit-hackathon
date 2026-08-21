import math
import re
from collections import Counter
from typing import Any

from .models import Evidence

TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣_]+")
CITE_PATTERN = re.compile(r"cite[_-]uid[\"\s:=]+[\"']?([A-Za-z0-9_-]+)", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text.replace("_", " ")) if len(token) > 1]


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
    if not documents:
        return []
    tokenized = [_tokens(passage), *(_tokens(document) for document in documents)]
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
    for content, tokens in zip(documents, tokenized[1:], strict=True):
        doc_vector = vector(tokens)
        doc_norm = math.sqrt(sum(value * value for value in doc_vector.values())) or 1
        score = sum(query_vector.get(token, 0) * value for token, value in doc_vector.items())
        score /= query_norm * doc_norm
        cite_uids = CITE_PATTERN.findall(content)
        if cite_uids:
            ranked.append(
                Evidence(
                    cite_uid=cite_uids[0],
                    relevance_score=max(0, min(1, score)),
                    tfidf_score=score,
                    content=content,
                )
            )
    ranked.sort(key=lambda item: item.tfidf_score, reverse=True)
    return ranked[:top_k]
