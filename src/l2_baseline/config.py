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
    max_retrieval_calls: int = Field(5, alias="L2_MAX_RETRIEVAL_CALLS", ge=1, le=30)
    max_reflection_rounds: int = Field(4, alias="L2_MAX_REFLECTION_ROUNDS", ge=1, le=10)
    retrieval_top_k: int = Field(3, alias="L2_RETRIEVAL_TOP_K", ge=1, le=10)
    tool_candidate_limit: int = Field(3, alias="L2_TOOL_CANDIDATE_LIMIT", ge=2, le=20)
    request_timeout_sec: float = Field(90, alias="L2_REQUEST_TIMEOUT_SEC", ge=10)

    @property
    def token(self) -> str:
        return self.api_key.get_secret_value()
