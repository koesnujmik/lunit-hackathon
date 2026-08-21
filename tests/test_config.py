from l2_baseline.config import Settings


def test_submission_has_embedded_api_key_fallback(monkeypatch) -> None:
    monkeypatch.delenv("LUNIT_FM_API_KEY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.token.startswith("lunit_")
    assert settings.token != "lunit_replace_me"
    assert settings.api_url == "https://model.hackathon.lunit.io"
    assert settings.model == "Lunit/L2-preview"
    assert settings.mcp_url == "https://mcp.hackathon.lunit.io/mcp"
    assert settings.patient_api_url == "https://patient.hackathon.lunit.io"
    assert settings.max_retrieval_calls == 5
    assert not hasattr(settings, "max_reflection_rounds")
    assert settings.tool_candidate_limit == 5
    assert settings.request_timeout_sec == 90


def test_environment_api_key_overrides_embedded_fallback(monkeypatch) -> None:
    monkeypatch.setenv("LUNIT_FM_API_KEY", "lunit_environment_override")

    settings = Settings(_env_file=None)

    assert settings.token == "lunit_environment_override"
