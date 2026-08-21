import json
import re

EXPLICIT_QUESTION_REQUEST_PATTERN = re.compile(
    r"\b(?:ask me (?:some )?questions?|ask (?:a |any )?clarifying questions?|"
    r"what (?:else )?do you need to know)\b|"
    r"(?:추가|확인|필요한|몇 가지)?\s*질문(?:을|도)?\s*(?:해|해줘|해주세요|해주실|하셔도)|"
    r"뭘\s*(?:더\s*)?(?:알아야|말해야)",
    re.IGNORECASE,
)
UNCERTAIN_DIAGNOSIS_PATTERN = re.compile(
    r"\b(?:one definite cause|definite (?:cause|diagnosis)|exact diagnosis|"
    r"no ifs or maybes|what (?:could|might) (?:this|it|these symptoms) be|"
    r"what do i have|what is (?:most likely )?causing (?:this|it|my))\b|"
    r"(?:딱|단)\s*하나의?\s*원인|확실한\s*원인|정확한\s*진단|"
    r"무슨\s*병|어떤\s*질환|원인이\s*(?:뭐|무엇)|왜\s*이러",
    re.IGNORECASE,
)
CHILD_PATTERN = re.compile(
    r"\b(?:child|kid|son|daughter|baby|infant|toddler|boy|girl)\b|"
    r"아이|아기|소아|아들|딸|어린이",
    re.IGNORECASE,
)
MEDICATION_PATTERN = re.compile(
    r"\b(?:medicine|medication|drug|dose|pill|tablet|syrup|acetaminophen|"
    r"paracetamol|ibuprofen|codeine)\b|"
    r"약|복용|먹여|투여|용량|알약|시럽|해열제|진통제",
    re.IGNORECASE,
)
MISSING_DETAIL_PATTERN = re.compile(
    r"\b(?:do not know|don't know|dont know|not sure|unsure|unknown|no clue|"
    r"approximately|around|probably|leftover|old bottle)\b|"
    r"모르|잘\s*모르|확실하지|대략|정도|남은\s*약|오래된",
    re.IGNORECASE,
)
NO_FOLLOW_UP_REQUEST_PATTERN = re.compile(
    r"\b(?:do not|don't|dont) ask (?:me )?(?:follow[- ]?up )?questions?\b|"
    r"\bwithout (?:any )?(?:follow[- ]?up )?questions?\b|"
    r"(?:추가\s*)?질문(?:은|을)?\s*(?:하지\s*마|하지\s*말|없이)",
    re.IGNORECASE,
)
IMMEDIATE_EMERGENCY_PATTERN = re.compile(
    r"\b(?:unresponsive|unconscious|not breathing|stopped breathing|"
    r"cannot breathe|can't breathe|severe trouble breathing|choking|"
    r"cardiac arrest)\b|"
    r"의식(?:이|을)?\s*(?:없|잃)|숨(?:을)?\s*(?:못\s*쉬|안\s*쉬)|"
    r"호흡(?:이)?\s*(?:없|멈)",
    re.IGNORECASE,
)
TERMINAL_QUESTION_PATTERN = re.compile(r"[?？](?:\s|[*_`\"'”’)}\]])*$")
FOLLOW_UP_DETAIL_PATTERNS = {
    "pediatric_medication": re.compile(
        r"\b(?:age|weight|ingredient|strength|breathing|sleepiness)\b|"
        r"나이|체중|성분|함량|호흡|처짐",
        re.IGNORECASE,
    ),
    "requested_questions": re.compile(
        r"\b(?:when|where|worse|swelling|warmth|fever)\b|"
        r"언제|부위|악화|붓기|열감|발열",
        re.IGNORECASE,
    ),
    "diagnostic_uncertainty": re.compile(
        r"\b(?:when|pain|swelling|injury|trauma|locking|giving way)\b|"
        r"언제|통증|붓기|외상|잠김|힘이\s*빠",
        re.IGNORECASE,
    ),
}
CLAIM_FRAMING_PATTERN = re.compile(
    r"\b(?:i heard|is (?:it|that|this) true|claim(?:ed)?|myth|really|"
    r"can|could|does|do)\b|"
    r"들었|사실(?:이야|인가|인가요)|정말|속설|주장",
    re.IGNORECASE,
)
CLAIM_EFFECT_PATTERN = re.compile(
    r"\b(?:cure|treat|heal|reverse|regrow|bring back hair|prevent|detox|"
    r"cause|improve|work for|effective for)\b|"
    r"치료|낫게|완치|되돌|머리카락|모발|예방|해독|원인|효과",
    re.IGNORECASE,
)
SAFETY_PERMISSION_PATTERN = re.compile(
    r"\b(?:can|should|may) i (?:give|take|use)|is it safe (?:to|if)|"
    r"safe (?:for|during)\b|"
    r"(?:먹|복용|투여|사용|줘|주어)도\s*(?:돼|되|괜찮)|안전(?:한가|할까|해)",
    re.IGNORECASE,
)
EVIDENCE_CALIBRATION_PATTERN = re.compile(
    r"\b(?:evidence|research|stud(?:y|ies)|data|support(?:s|ed)?|"
    r"shown?|demonstrat(?:e|es|ed)|proven|established)\b|"
    r"근거|연구|자료|뒷받침|입증|확인",
    re.IGNORECASE,
)
NEGATIVE_ANSWER_PATTERN = re.compile(
    r"^\s*(?:[*_`]+)?(?:no|아니요|아니오)\b",
    re.IGNORECASE,
)
BARE_NEGATIVE_ANSWER_PATTERN = re.compile(
    r"^\s*(?:[*_`]+)?(?:no|아니요|아니오)(?:[*_`]+)?[.!。！]?\s*$",
    re.IGNORECASE,
)


def _conversation_context(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"{message.get('role', 'user').upper()}: {message.get('content', '')}"
        for message in messages
    )


def _latest_user_text(messages: list[dict[str, str]]) -> str:
    return next(
        (
            message.get("content", "")
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )


def _log(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def _follow_up_requirement(messages: list[dict[str, str]]) -> str | None:
    """Identify narrow cases where an explicit user-facing question is essential."""
    context = _conversation_context(messages)
    latest = _latest_user_text(messages)
    if NO_FOLLOW_UP_REQUEST_PATTERN.search(latest):
        return None
    if IMMEDIATE_EMERGENCY_PATTERN.search(latest):
        return None
    if (
        CHILD_PATTERN.search(context)
        and MEDICATION_PATTERN.search(context)
        and (
            MISSING_DETAIL_PATTERN.search(context)
            or not re.search(
                r"\b\d+(?:\.\d+)?\s*(?:kg|kilograms?|lb|lbs|pounds?)\b",
                context,
                re.IGNORECASE,
            )
        )
    ):
        return "pediatric_medication"
    if EXPLICIT_QUESTION_REQUEST_PATTERN.search(context):
        return "requested_questions"
    if UNCERTAIN_DIAGNOSIS_PATTERN.search(latest):
        return "diagnostic_uncertainty"
    return None


def _requires_claim_calibration(messages: list[dict[str, str]]) -> bool:
    """Detect efficacy or causation claims without weakening safety prohibitions."""
    context = _conversation_context(messages)
    if SAFETY_PERMISSION_PATTERN.search(context):
        return False
    return bool(
        CLAIM_FRAMING_PATTERN.search(context) and CLAIM_EFFECT_PATTERN.search(context)
    )


def _ensure_claim_calibration(answer: str, required: bool, latest: str) -> str:
    """Qualify unsupported health claims without adding another model call."""
    if not required:
        return answer
    if EVIDENCE_CALIBRATION_PATTERN.search(answer):
        _log("claim_calibration_audit_completed", passed=True, issues=[])
        return answer

    korean = any("가" <= character <= "힣" for character in latest)
    negative = bool(NEGATIVE_ANSWER_PATTERN.search(answer))
    if negative:
        calibration = (
            "아니요. 현재 신뢰할 만한 의학적 근거는 그 주장을 뒷받침하지 않습니다."
            if korean
            else "No—current reliable medical evidence does not support that claim."
        )
    else:
        calibration = (
            "이 결론의 확실성은 현재 신뢰할 만한 의학적 근거의 강도에 맞춰 "
            "해석해야 합니다."
            if korean
            else "The certainty of this conclusion should match the strength of current "
            "reliable medical evidence."
        )
    _log(
        "claim_calibration_audit_completed",
        passed=False,
        issues=["missing_evidence_calibration"],
    )
    repaired = (
        calibration
        if negative and BARE_NEGATIVE_ANSWER_PATTERN.fullmatch(answer)
        else answer.rstrip() + "\n\n" + calibration
    )
    _log("claim_calibration_repaired", negative_answer=negative)
    return repaired


def _follow_up_audit(answer: str, requirement: str | None) -> list[str]:
    """Check that a required follow-up is specific and placed at the end."""
    if requirement is None:
        return []
    stripped = answer.strip()
    final_paragraph = re.split(r"\n\s*\n", stripped)[-1] if stripped else ""
    issues: list[str] = []
    if not TERMINAL_QUESTION_PATTERN.search(final_paragraph):
        issues.append("missing_terminal_question")
    question_count = len(re.findall(r"[?？]", final_paragraph))
    if question_count < 1 or question_count > 2:
        issues.append("terminal_question_count")
    detail_pattern = FOLLOW_UP_DETAIL_PATTERNS.get(requirement)
    if detail_pattern is not None and not detail_pattern.search(final_paragraph):
        issues.append("non_specific_terminal_question")
    return issues


def _ensure_required_follow_up(answer: str, requirement: str | None, latest: str) -> str:
    """Add one bounded, high-yield question without another model call."""
    issues = _follow_up_audit(answer, requirement)
    if not issues:
        if requirement is not None:
            _log("follow_up_audit_completed", passed=True, issues=[])
        return answer
    korean = any("가" <= character <= "힣" for character in latest)
    if requirement == "pediatric_medication":
        question = (
            "안전성을 더 정확히 판단하려면 아이의 정확한 나이와 현재 체중, 약의 "
            "성분명·함량, 호흡곤란이나 심한 처짐 여부를 알려주시겠어요?"
            if korean
            else "To tailor this safely, what are your child's exact age and current weight, "
            "the medicine's active ingredient and strength, and whether there is trouble "
            "breathing or unusual sleepiness?"
        )
    elif requirement == "requested_questions":
        question = (
            "판단에 가장 도움이 되도록, 증상이 언제 시작됐고 가장 심한 부위와 "
            "악화 요인, 붓기·열감·발열 같은 동반 증상이 있는지 알려주시겠어요?"
            if korean
            else "To narrow this down, when did the symptoms start, where are they worst, "
            "what makes them worse, and is there swelling, warmth, or fever?"
        )
    else:
        question = (
            "가능성을 더 좁히려면 증상이 언제 시작됐고 통증·붓기·외상·잠김 또는 "
            "힘이 빠지는 증상이 있는지 알려주시겠어요?"
            if korean
            else "To narrow this down, when did it start, and is there pain, swelling, an "
            "injury, locking, or giving way?"
        )
    _log("follow_up_audit_completed", passed=False, issues=issues)
    repaired = answer.rstrip() + "\n\n" + question
    remaining_issues = _follow_up_audit(repaired, requirement)
    _log(
        "follow_up_question_appended",
        reason=requirement,
        passed=not remaining_issues,
        issues=remaining_issues,
    )
    return repaired
