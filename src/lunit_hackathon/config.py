from __future__ import annotations

import os
from dataclasses import dataclass


def _clean_base_url(value: str) -> str:
    return value.strip().rstrip("/")


@dataclass(frozen=True)
class Settings:
    fm_api_url: str
    fm_api_key: str
    fm_model: str
    mcp_url: str
    patient_api_url: str
    patient_model: str
    timeout_sec: float
    enable_retrieval: bool
    routing_mode: str
    max_retrieval_tool_calls: int
    max_tool_result_chars: int
    max_evidence_chars: int


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean, got {raw!r}.")


def _read_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}.") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be at least 1, got {value}.")
    return value


def load_settings(require_api_key: bool = True) -> Settings:
    api_key = os.getenv("LUNIT_FM_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise RuntimeError("Missing LUNIT_FM_API_KEY. Set it in your shell or source .env first.")

    timeout_raw = os.getenv("LUNIT_TIMEOUT_SEC", "60").strip()
    try:
        timeout_sec = float(timeout_raw)
    except ValueError as exc:
        raise RuntimeError(f"LUNIT_TIMEOUT_SEC must be numeric, got {timeout_raw!r}.") from exc

    routing_mode = os.getenv("LUNIT_ROUTING_MODE", "hybrid").strip().lower()
    if routing_mode not in {"hybrid", "model"}:
        raise RuntimeError(
            f"LUNIT_ROUTING_MODE must be 'hybrid' or 'model', got {routing_mode!r}."
        )

    return Settings(
        fm_api_url=_clean_base_url(os.getenv("LUNIT_FM_API_URL", "https://model.hackathon.lunit.io")),
        fm_api_key=api_key,
        fm_model=os.getenv("LUNIT_FM_MODEL", "Lunit/L2-preview").strip(),
        mcp_url=_clean_base_url(os.getenv("LUNIT_MCP_URL", "https://mcp.hackathon.lunit.io/mcp")),
        patient_api_url=_clean_base_url(
            os.getenv("LUNIT_PATIENT_API_URL", "https://patient.hackathon.lunit.io")
        ),
        patient_model=os.getenv("LUNIT_PATIENT_MODEL", "patient-simulator-ko").strip(),
        timeout_sec=timeout_sec,
        enable_retrieval=_read_bool("LUNIT_ENABLE_RETRIEVAL", True),
        routing_mode=routing_mode,
        max_retrieval_tool_calls=_read_positive_int("LUNIT_MAX_RETRIEVAL_TOOL_CALLS", 6),
        max_tool_result_chars=_read_positive_int("LUNIT_MAX_TOOL_RESULT_CHARS", 12_000),
        max_evidence_chars=_read_positive_int("LUNIT_MAX_EVIDENCE_CHARS", 24_000),
    )
