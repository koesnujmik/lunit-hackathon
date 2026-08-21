from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch

from l2_baseline.config import Settings


def test_submission_key_file_is_used_without_runtime_key(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    key_file = tmp_path / "submission_api_key"
    key_file.write_text("lunit_file_key\n", encoding="utf-8")
    for name in ("LUNIT_FM_API_KEY", "OPENAI_API_KEY", "LUNIT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LUNIT_SUBMISSION_API_KEY_FILE", str(key_file))

    settings = Settings(_env_file=None)

    assert settings.token == "lunit_file_key"


def test_runtime_key_overrides_submission_key_file(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    key_file = tmp_path / "submission_api_key"
    key_file.write_text("lunit_file_key\n", encoding="utf-8")
    monkeypatch.setenv("LUNIT_FM_API_KEY", "lunit_runtime_key")
    monkeypatch.setenv("LUNIT_SUBMISSION_API_KEY_FILE", str(key_file))

    settings = Settings(_env_file=None)

    assert settings.token == "lunit_runtime_key"


def test_timeout_defaults_bound_one_turn(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("LUNIT_FM_API_KEY", "lunit_test")

    settings = Settings(_env_file=None)

    assert settings.request_timeout_sec == 45
    assert settings.turn_timeout_sec == 55
    assert settings.retrieval_timeout_sec == 18
    assert settings.max_retrieval_calls == 2
    assert settings.generation_max_tokens == 2_048
    assert settings.max_history_messages == 6
