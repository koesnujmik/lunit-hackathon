from l2_baseline.config import Settings


def test_submission_has_embedded_api_key_fallback(monkeypatch) -> None:
    monkeypatch.delenv("LUNIT_FM_API_KEY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.token.startswith("lunit_")
    assert settings.token != "lunit_replace_me"


def test_environment_api_key_overrides_embedded_fallback(monkeypatch) -> None:
    monkeypatch.setenv("LUNIT_FM_API_KEY", "lunit_environment_override")

    settings = Settings(_env_file=None)

    assert settings.token == "lunit_environment_override"
