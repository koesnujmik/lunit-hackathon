from __future__ import annotations

from typing import Any


def source_families(text: str) -> set[str]:
    normalized = text.lower()
    families: set[str] = set()

    def contains(*terms: str) -> bool:
        return any(term in normalized for term in terms)

    if contains("guideline", "guidance", "가이드라인", "진료지침"):
        families.add("guideline")
    if contains("hira", "reimbursement", "coverage", "급여", "심평원", "고시", "삭감"):
        families.add("hira")
    if contains(
        "dailymed",
        "mfds",
        "drug label",
        "drug",
        "medication",
        "dose",
        "dosage",
        "interaction",
        "contraindication",
        "의약품",
        "약물",
        "약제",
        "약은",
        "약을",
        "약의",
        "약과",
        "약이",
        "처방약",
        "복용",
        "투여",
        "용량",
        "상호작용",
        "금기",
        "허가사항",
    ):
        families.add("drug")
    if contains(
        "law",
        "statute",
        "regulation",
        "법률",
        "법령",
        "법적",
        "의료법",
        "시행령",
        "조문",
    ):
        families.add("law")
    if contains("kcd", "diagnosis code", "disease code", "상병", "질병코드", "진단코드"):
        families.add("code")
    if contains(
        "pubmed",
        "study",
        "studies",
        "trial",
        "literature",
        "paper",
        "연구",
        "논문",
        "문헌",
    ):
        families.add("research")
    if contains("faers", "adverse event", "pharmacovigilance", "부작용 신고", "이상사례"):
        families.add("faers")
    return families


def explicitly_requests_retrieval(conversation: list[dict[str, Any]]) -> bool:
    latest_user = latest_user_text(conversation).lower()
    return any(
        term in latest_user
        for term in (
            "guideline",
            "가이드라인",
            "진료지침",
            "source",
            "citation",
            "출처",
            "근거",
            "evidence",
            "latest",
            "current",
            "최신",
            "현행",
            "official",
            "공식",
            "hira",
            "심평원",
            "mfds",
            "식약처",
            "dailymed",
            "pubmed",
            "faers",
            "kcd",
            "급여",
            "허가정보",
            "법령",
            "조문",
            "논문",
            "문헌",
        )
    )


def latest_user_text(conversation: list[dict[str, Any]]) -> str:
    for message in reversed(conversation):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def conversation_user_text(conversation: list[dict[str, Any]]) -> str:
    return "\n".join(
        message["content"]
        for message in conversation
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    )
