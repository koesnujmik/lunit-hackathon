import math
import re
from collections import Counter
from typing import Any

from .models import Evidence

TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣_]+")
CITE_PATTERN = re.compile(r"cite[_-]uid[\"\s:=]+[\"']?([A-Za-z0-9_-]+)", re.IGNORECASE)
BM25_K1 = 1.5
BM25_B = 0.75
QUERY_WEIGHT = 0.65
RATIONALE_WEIGHT = 0.35
ABSOLUTE_SCORE_CUTOFF = 0.15
RELATIVE_SCORE_CUTOFF = 0.35
MIN_EVIDENCE_CANDIDATES = 2
MAX_EVIDENCE_CANDIDATES = 5


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text.replace("_", " ")) if len(token) > 1]


def _bm25_scores(
    query_tokens: list[str], document_tokens: list[list[str]]
) -> list[float]:
    """Return non-negative Okapi BM25 scores for an in-memory candidate set."""
    if not document_tokens:
        return []
    document_count = len(document_tokens)
    average_length = (
        sum(len(tokens) for tokens in document_tokens) / document_count or 1.0
    )
    query_terms = set(query_tokens)
    document_frequency = Counter(
        token
        for tokens in document_tokens
        for token in query_terms.intersection(tokens)
    )
    inverse_document_frequency = {
        token: math.log(
            1
            + (document_count - frequency + 0.5)
            / (frequency + 0.5)
        )
        for token, frequency in document_frequency.items()
    }

    scores: list[float] = []
    for tokens in document_tokens:
        frequencies = Counter(tokens)
        length_normalization = BM25_K1 * (
            1 - BM25_B + BM25_B * len(tokens) / average_length
        )
        score = 0.0
        for token in query_terms:
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            score += inverse_document_frequency[token] * (
                frequency * (BM25_K1 + 1)
                / (frequency + length_normalization)
            )
        scores.append(score)
    return scores


def _normalize_scores(scores: list[float]) -> list[float]:
    maximum = max(scores, default=0.0)
    if maximum <= 0:
        return [0.0 for _ in scores]
    return [score / maximum for score in scores]


def _hybrid_bm25_scores(
    query: str, rationale: str, document_tokens: list[list[str]]
) -> list[float]:
    query_scores = _normalize_scores(_bm25_scores(_tokens(query), document_tokens))
    if not rationale.strip():
        return query_scores
    rationale_scores = _normalize_scores(
        _bm25_scores(_tokens(rationale), document_tokens)
    )
    return [
        QUERY_WEIGHT * query_score + RATIONALE_WEIGHT * rationale_score
        for query_score, rationale_score in zip(
            query_scores, rationale_scores, strict=True
        )
    ]


def rank_tool_candidates(
    search_text: str,
    tools: list[dict[str, Any]],
    limit: int = 6,
    rationale: str = "",
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
    tokenized_tools = [_tokens(text) for text in tool_texts]
    scores = _hybrid_bm25_scores(search_text, rationale, tokenized_tools)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, (tool, score) in enumerate(zip(tools, scores, strict=True)):
        scored.append((score, -index, tool))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:limit]]


def rank_documents(query: str, rationale: str, documents: list[str]) -> list[Evidence]:
    """Rank citable MCP results and retain an adaptive evidence candidate set."""
    if not documents:
        return []
    tokenized_documents = [_tokens(document) for document in documents]
    scores = _hybrid_bm25_scores(query, rationale, tokenized_documents)
    ranked: list[Evidence] = []
    for content, score in zip(documents, scores, strict=True):
        cite_uids = CITE_PATTERN.findall(content)
        if cite_uids:
            ranked.append(
                Evidence(
                    cite_uid=cite_uids[0],
                    relevance_score=score,
                    bm25_score=score,
                    content=content,
                )
            )
    ranked.sort(key=lambda item: item.bm25_score, reverse=True)
    if not ranked:
        return []

    cutoff = max(
        ABSOLUTE_SCORE_CUTOFF,
        ranked[0].bm25_score * RELATIVE_SCORE_CUTOFF,
    )
    passing_count = sum(item.bm25_score >= cutoff for item in ranked)
    candidate_count = min(
        MAX_EVIDENCE_CANDIDATES,
        max(MIN_EVIDENCE_CANDIDATES, passing_count),
        len(ranked),
    )
    return ranked[:candidate_count]
