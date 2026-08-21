from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    fm_api_url: str
    fm_api_key: str
    fm_model: str
    patient_api_url: str
    patient_model: str
    timeout_sec: float
    max_tokens: int


def load_settings(require_api_key: bool = True) -> Settings:
    api_key = os.getenv("LUNIT_FM_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise RuntimeError("Missing LUNIT_FM_API_KEY.")

    return Settings(
        fm_api_url=os.getenv(
            "LUNIT_FM_API_URL", "https://model.hackathon.lunit.io"
        ).rstrip("/"),
        fm_api_key=api_key,
        fm_model=os.getenv("LUNIT_FM_MODEL", "Lunit/L2-preview").strip(),
        patient_api_url=os.getenv(
            "LUNIT_PATIENT_API_URL", "https://patient.hackathon.lunit.io"
        ).rstrip("/"),
        patient_model=os.getenv("LUNIT_PATIENT_MODEL", "patient-simulator-ko").strip(),
        timeout_sec=float(os.getenv("LUNIT_TIMEOUT_SEC", "40")),
        max_tokens=min(2_048, int(os.getenv("LUNIT_MAX_TOKENS", "2048"))),
    )
