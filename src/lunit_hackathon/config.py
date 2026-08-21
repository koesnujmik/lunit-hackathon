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
    patient_api_url: str
    patient_model: str
    timeout_sec: float


def load_settings(require_api_key: bool = True) -> Settings:
    api_key = os.getenv("LUNIT_FM_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise RuntimeError("Missing LUNIT_FM_API_KEY. Set it in your shell or source .env first.")

    timeout_raw = os.getenv("LUNIT_TIMEOUT_SEC", "60").strip()
    try:
        timeout_sec = float(timeout_raw)
    except ValueError as exc:
        raise RuntimeError(f"LUNIT_TIMEOUT_SEC must be numeric, got {timeout_raw!r}.") from exc

    return Settings(
        fm_api_url=_clean_base_url(os.getenv("LUNIT_FM_API_URL", "https://model.hackathon.lunit.io")),
        fm_api_key=api_key,
        fm_model=os.getenv("LUNIT_FM_MODEL", "Lunit/L2-preview").strip(),
        patient_api_url=_clean_base_url(
            os.getenv("LUNIT_PATIENT_API_URL", "https://patient.hackathon.lunit.io")
        ),
        patient_model=os.getenv("LUNIT_PATIENT_MODEL", "patient-simulator-ko").strip(),
        timeout_sec=timeout_sec,
    )
