from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_url: str = Field("https://model.hackathon.lunit.io", alias="LUNIT_FM_API_URL")
    # Submission fallback: the evaluator does not mount the local .env file.
    # An injected LUNIT_FM_API_KEY still takes precedence over this value.
    api_key: SecretStr = Field(
        default=SecretStr("lunit_E2V91QFRZ1pr54qdcWmmuHigpP68ZRHA9fgIts31ocY"),
        alias="LUNIT_FM_API_KEY",
    )
    model: str = Field("Lunit/L2-preview", alias="LUNIT_FM_MODEL")
    mcp_url: str = Field("https://mcp.hackathon.lunit.io/mcp", alias="LUNIT_MCP_URL")
    patient_api_url: str = Field(
        "https://patient.hackathon.lunit.io", alias="LUNIT_PATIENT_API_URL"
    )
    max_retrieval_calls: int = Field(4, alias="L2_MAX_RETRIEVAL_CALLS", ge=1, le=8)
    max_retrieval_rounds: int = Field(3, alias="L2_MAX_RETRIEVAL_ROUNDS", ge=1, le=4)
    retrieval_top_k: int = Field(3, alias="L2_RETRIEVAL_TOP_K", ge=1, le=10)
    request_timeout_sec: float = Field(45, alias="L2_REQUEST_TIMEOUT_SEC", ge=10, le=120)
    retrieval_timeout_sec: float = Field(45, alias="L2_RETRIEVAL_TIMEOUT_SEC", ge=10, le=90)
    max_tool_result_chars: int = Field(
        8_000, alias="L2_MAX_TOOL_RESULT_CHARS", ge=1_000, le=30_000
    )
    max_evidence_chars: int = Field(
        16_000, alias="L2_MAX_EVIDENCE_CHARS", ge=2_000, le=40_000
    )
    retrieval_max_tokens: int = Field(
        768, alias="L2_RETRIEVAL_MAX_TOKENS", ge=256, le=2_048
    )
    generation_max_tokens: int = Field(
        6_144, alias="L2_GENERATION_MAX_TOKENS", ge=512, le=8_192
    )

    @property
    def token(self) -> str:
        return self.api_key.get_secret_value()
