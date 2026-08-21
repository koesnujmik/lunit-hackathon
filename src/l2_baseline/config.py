import os
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SUBMISSION_API_KEY_FILE = Path("submission_api_key")


def _load_submission_api_key() -> SecretStr:
    path = Path(
        os.getenv(
            "LUNIT_SUBMISSION_API_KEY_FILE",
            str(DEFAULT_SUBMISSION_API_KEY_FILE),
        )
    )
    try:
        api_key = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(
            "No runtime Lunit API key was supplied and the submission key "
            f"file could not be read: {path}"
        ) from exc

    if not api_key.startswith("lunit_"):
        raise ValueError("submission_api_key must contain one lunit_ API key")
    return SecretStr(api_key)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_url: str = Field("https://model.hackathon.lunit.io", alias="LUNIT_FM_API_URL")
    api_key: SecretStr = Field(
        default_factory=_load_submission_api_key,
        validation_alias=AliasChoices(
            "LUNIT_FM_API_KEY", "OPENAI_API_KEY", "LUNIT_API_KEY"
        ),
    )
    model: str = Field("Lunit/L2-preview", alias="LUNIT_FM_MODEL")
    mcp_url: str = Field("https://mcp.hackathon.lunit.io/mcp", alias="LUNIT_MCP_URL")
    patient_api_url: str = Field(
        "https://patient.hackathon.lunit.io", alias="LUNIT_PATIENT_API_URL"
    )
    max_retrieval_calls: int = Field(4, alias="L2_MAX_RETRIEVAL_CALLS", ge=1, le=4)
    retrieval_top_k: int = Field(3, alias="L2_RETRIEVAL_TOP_K", ge=1, le=10)
    tool_candidate_limit: int = Field(8, alias="L2_TOOL_CANDIDATE_LIMIT", ge=2, le=20)
    retrieval_timeout_sec: float = Field(
        25, alias="L2_RETRIEVAL_TIMEOUT_SEC", ge=5, le=25
    )
    max_tool_result_chars: int = Field(
        6_000, alias="L2_MAX_TOOL_RESULT_CHARS", ge=1_000, le=12_000
    )
    max_evidence_chars: int = Field(
        10_000, alias="L2_MAX_EVIDENCE_CHARS", ge=2_000, le=20_000
    )
    retrieval_max_tokens: int = Field(
        512, alias="L2_RETRIEVAL_MAX_TOKENS", ge=256, le=1_024
    )
    request_timeout_sec: float = Field(
        45, alias="L2_REQUEST_TIMEOUT_SEC", ge=10, le=50
    )
    turn_timeout_sec: float = Field(55, alias="L2_TURN_TIMEOUT_SEC", ge=15, le=60)
    generation_max_tokens: int = Field(
        2_048, alias="L2_GENERATION_MAX_TOKENS", ge=256, le=2_048
    )
    max_history_messages: int = Field(6, alias="L2_MAX_HISTORY_MESSAGES", ge=2, le=10)
    max_message_chars: int = Field(4_000, alias="L2_MAX_MESSAGE_CHARS", ge=500, le=10_000)

    @property
    def token(self) -> str:
        return self.api_key.get_secret_value()
