from pathlib import Path

from shu_insight.settings import AppSettings


def test_settings_uses_project_local_database(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SHU_NOTICE_CONTACT", raising=False)
    monkeypatch.delenv("SHU_NOTICE_AI_ENABLED", raising=False)

    settings = AppSettings.load(tmp_path)

    assert settings.database_path == tmp_path / "data" / "shu_insight.sqlite3"
    assert settings.ai_enabled is False


def test_contact_is_optional_default_without_enabling_ai(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SHU_NOTICE_CONTACT", "student@example.com")
    monkeypatch.delenv("SHU_NOTICE_AI_ENABLED", raising=False)

    settings = AppSettings.load(tmp_path)

    assert settings.contact == "student@example.com"
    assert settings.ai_enabled is False


def test_database_path_can_be_overridden_for_isolated_runs(
    tmp_path: Path, monkeypatch
) -> None:
    custom_path = tmp_path / "isolated" / "demo.sqlite3"
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(custom_path))

    settings = AppSettings.load(tmp_path / "project")

    assert settings.database_path == custom_path.resolve()


def test_session_contact_overrides_env_without_changing_persistent_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SHU_NOTICE_CONTACT", "environment@example.invalid")
    settings = AppSettings.load(tmp_path)
    assert settings.effective_contact("  session@example.invalid  ") == "session@example.invalid"
    assert settings.effective_contact("") == ""
    assert settings.effective_contact("   ") == ""
    assert settings.effective_contact() == "environment@example.invalid"
    assert settings.contact == "environment@example.invalid"
    assert AppSettings.load(tmp_path).contact == "environment@example.invalid"
