import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .models import RetrievalPipeline

PIPELINE_ROOT_TOOLS: dict[RetrievalPipeline, tuple[str, ...]] = {
    "direct": (),
    "index": ("index_get_relevant_nodes", "index_keyword_search"),
    "law": ("openapi_law_search",),
    "rag_sql": ("rag_get_data_source_detail",),
    "rag_vector": ("rag_get_data_source_detail",),
    "drug_label": (
        "openapi_mfds_get_drug_indication",
        "openapi_mfds_check_drug_permission",
    ),
    "drug_substitution": (
        "openapi_mfds_get_drug_indication",
        "openapi_mfds_check_drug_permission",
    ),
    "kcd_billing": ("kcd_search_codes",),
}


PIPELINE_HINTS: dict[RetrievalPipeline, str] = {
    "direct": "Use the normal direct retrieval selector.",
    "index": (
        "Choose semantic node search, exact keyword search, or both. Set corpus_tag to "
        "guideline or hira. These discovery results will deterministically feed page retrieval."
    ),
    "law": (
        "Search by the Korean law name or a broad law-name topic. The returned MST will feed "
        "article-list and article-text stages."
    ),
    "rag_sql": (
        "Choose exactly one SQL source detail: faers_12q4_25q4, dailymed_26_08, or kcd. "
        "Prefer FAERS only for adverse-event report analysis."
    ),
    "rag_vector": (
        "Choose exactly one vector source detail: pubmed_abstracts or hira_faq."
    ),
    "drug_label": (
        "Resolve the Korean product using an MFDS tool so ingredient_eng can feed the English "
        "DailyMed label lookup. Prefer get_drug_indication when either MFDS tool is adequate."
    ),
    "drug_substitution": (
        "Resolve the Korean product using an MFDS tool so its English ingredient can feed a "
        "same-ingredient product search."
    ),
    "kcd_billing": (
        "Search the disease name for candidate KCD codes. Exact-code validation happens in the "
        "following fixed stage."
    ),
}


def select_pipeline_type(query: str) -> RetrievalPipeline:
    """Route explicit source/query families to fixed pipelines without an LLM decision."""
    text = query.casefold()
    has_hangul = bool(re.search(r"[가-힣]", query))

    if re.search(r"동일\s*성분|대체약|대체\s*의약품|same[- ]ingredient|substitut", text):
        return "drug_substitution"
    if "kcd" in text and re.search(r"청구|상병|코드|billing|billable|disease code", text):
        return "kcd_billing"
    if re.search(
        r"법률|법령|시행령|시행규칙|조문|법적|법상|법에\s*따라|"
        r"법\s*제\s*\d+\s*조|법(?:에서|에\s*따르면|의)|"
        r"\b(?:law|statute|legal|act)\b|\barticle\s+\d+",
        text,
    ):
        return "law"
    if has_hangul and re.search(
        r"dailymed|fda(?:-approved)?\s+label|official\s+label|공식\s*라벨|미국\s*라벨",
        text,
    ):
        return "drug_label"
    if re.search(
        r"\bfaers\b|adverse event reporting system|reporting odds ratio|"
        r"disproportionality|\bprr\b",
        text,
    ):
        return "rag_sql"
    if re.search(r"\bpubmed\b|\bpmc\b|hira\s*faq|심평원\s*faq|문헌\s*검색", text):
        return "rag_vector"
    if re.search(
        r"guidelines?|clinical practice|가이드라인|진료\s*지침|임상\s*지침|"
        r"권고안|학회\s*(?:지침|권고)|급여\s*기준|심평원\s*(?:고시|공고)|"
        r"\b(?:kdigo|nccn|esmo|asco)\b|\bacc\s*/\s*aha\b",
        text,
    ):
        return "index"
    return "direct"


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolAction:
    function: FunctionSpec


@dataclass(frozen=True)
class PageTarget:
    corpus_tag: str
    doc_id: str
    start_page: int
    end_page: int


def make_action(name: str, arguments: dict[str, Any]) -> ToolAction:
    return ToolAction(
        function=FunctionSpec(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        )
    )


def tools_by_name(
    tools: list[dict[str, Any]], names: Iterable[str]
) -> list[dict[str, Any]]:
    available = {tool["function"]["name"]: tool for tool in tools}
    return [available[name] for name in names if name in available]


def _payload(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped or stripped.startswith("MCP tool error:"):
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for marker in ("[", "{"):
            start = stripped.find(marker)
            if start < 0:
                continue
            try:
                return decoder.raw_decode(stripped[start:])[0]
            except json.JSONDecodeError:
                continue
    return None


def _mappings(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _mappings(child)


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def values_for_keys(texts: Iterable[str], keys: set[str]) -> list[str]:
    lowered = {key.lower() for key in keys}
    values: list[str] = []
    for text in texts:
        payload = _payload(text)
        if payload is not None:
            for mapping in _mappings(payload):
                for key, value in mapping.items():
                    if key.lower() not in lowered:
                        continue
                    if isinstance(value, (str, int, float)):
                        values.append(str(value))
                    elif isinstance(value, list):
                        values.extend(str(item) for item in value if isinstance(item, str))
        for key in keys:
            pattern = rf"(?i){re.escape(key)}\s*[:=]\s*[\"']?([A-Za-z0-9_.:-]+)"
            values.extend(re.findall(pattern, text))
    return _unique(values)


def identifier_entries(texts: Iterable[str], key_name: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    pattern = re.compile(
        rf"(?i){re.escape(key_name)}\s*[:=]\s*[\"']?([A-Za-z0-9_.:-]+)"
    )
    for text in texts:
        payload = _payload(text)
        sources: list[str] = []
        if payload is not None:
            sources.extend(_strings(payload))
            for mapping in _mappings(payload):
                value = next(
                    (
                        str(item)
                        for key, item in mapping.items()
                        if key.lower() == key_name.lower()
                        and isinstance(item, (str, int, float))
                    ),
                    "",
                )
                if value and value not in seen:
                    seen.add(value)
                    entries.append((value, json.dumps(mapping, ensure_ascii=False)))
        sources.append(text)
        for source in sources:
            for line in source.splitlines():
                match = pattern.search(line)
                if match and match.group(1) not in seen:
                    value = match.group(1)
                    seen.add(value)
                    entries.append((value, line.strip()))
    return entries


def page_targets(action_outputs: Iterable[tuple[Any, str]]) -> list[PageTarget]:
    targets: list[PageTarget] = []
    seen: set[tuple[str, str, int, int]] = set()
    for action, text in action_outputs:
        try:
            arguments = json.loads(action.function.arguments)
        except (AttributeError, json.JSONDecodeError, TypeError):
            continue
        corpus_tag = arguments.get("corpus_tag")
        payload = _payload(text)
        if not isinstance(corpus_tag, str) or payload is None:
            continue
        for mapping in _mappings(payload):
            doc_id = mapping.get("doc_id")
            page_range = mapping.get("range")
            if isinstance(page_range, list) and len(page_range) == 2:
                start_page, end_page = page_range
            elif isinstance(mapping.get("page"), int):
                start_page = end_page = mapping["page"]
            else:
                start_page = mapping.get("start_page")
                end_page = mapping.get("end_page")
            if not (
                isinstance(doc_id, str)
                and isinstance(start_page, int)
                and isinstance(end_page, int)
                and start_page > 0
                and end_page >= start_page
            ):
                continue
            end_page = min(end_page, start_page + 19)
            key = (corpus_tag, doc_id, start_page, end_page)
            if key not in seen:
                seen.add(key)
                targets.append(PageTarget(*key))
    return targets


def article_entries(texts: Iterable[str]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    key_names = {"article_key", "articlekey", "조문키"}
    for text in texts:
        payload = _payload(text)
        sources = [text]
        if payload is not None:
            sources.extend(_strings(payload))
            for mapping in _mappings(payload):
                key = next(
                    (
                        str(value)
                        for name, value in mapping.items()
                        if name.lower() in key_names and isinstance(value, (str, int))
                    ),
                    "",
                )
                if key and key not in seen:
                    seen.add(key)
                    entries.append((key, json.dumps(mapping, ensure_ascii=False)))
        for source in sources:
            for line in source.splitlines():
                match = re.match(r"^\s*([^\s,:]+)\s+제\s*\d+\s*조", line)
                if match and match.group(1) not in seen:
                    key = match.group(1)
                    seen.add(key)
                    entries.append((key, line.strip()))
    return entries


def english_ingredients(texts: Iterable[str]) -> list[str]:
    materialized = list(texts)
    candidates: list[str] = []
    for key in (
        "ingredient_eng",
        "ingredient_english",
        "item_ingr_name",
        "ingredient",
    ):
        candidates.extend(values_for_keys(materialized, {key}))
    return _unique(value for value in candidates if re.search(r"[A-Za-z]", value))


def product_names(texts: Iterable[str]) -> list[str]:
    return values_for_keys(
        texts,
        {
            "drug_name",
            "product_name",
            "item_name",
            "item_name_kor",
            "item_name_ko",
            "품목명",
            "품명",
        },
    )


def kcd_candidates(texts: Iterable[str]) -> list[tuple[str, str | None]]:
    candidates: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    for text in texts:
        payload = _payload(text)
        if payload is None:
            continue
        for mapping in _mappings(payload):
            code = next(
                (
                    str(value)
                    for key, value in mapping.items()
                    if key.lower() in {"code", "kcd_code", "disease_code"}
                    and isinstance(value, (str, int))
                ),
                "",
            )
            if not re.fullmatch(
                r"[A-Z][0-9]{2}(?:\.[0-9A-Z]+)?", code, re.IGNORECASE
            ):
                continue
            revision_value = next(
                (
                    str(value)
                    for key, value in mapping.items()
                    if key.lower() in {"revision", "version", "kcd_version"}
                    and isinstance(value, (str, int))
                ),
                None,
            )
            candidate = (code, revision_value)
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates
