from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository
from shu_insight.db.schema import Database
from shu_insight.ingest.pipeline import SourceUpdateResult, UpdateEvent, UpdateReport
from shu_insight.ingest.preload import import_preload
from shu_insight.services.update import UpdateService
from shu_insight.sources.registry import load_source_registry

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def app(tmp_path, monkeypatch):
    database = Database(tmp_path / "app.sqlite3")
    database.initialize()
    registry = load_source_registry(ROOT / "config/sources.toml")
    repository = NoticeRepository(database, OfficialNoticePolicy({
        key: adapter.config for key, adapter in registry.items()
    }))
    import_preload(ROOT / "data/official/preload-2026-10-25.json", repository)
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(database.path))
    monkeypatch.setenv("SHU_NOTICE_AI_ENABLED", "false")
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)


def test_sidebar_stock_counts_are_unfiltered_and_include_all_four_sources(app):
    app.run()
    assert not app.exception
    assert "收录统计" in [item.value for item in app.sidebar.header]
    counts = [item.value for item in app.sidebar.markdown if " 条**" in item.value]
    assert counts == [
        "上海大学官网：**25 条**", "本科生院：**42 条**",
        "研究生院：**5 条**", "计算机学院：**2 条**",
    ]
    app.sidebar.text_input(key="search-keywords").set_value("不存在的搜索ZXQ927").run()
    assert not app.exception
    assert [item.value for item in app.sidebar.markdown if " 条**" in item.value] == counts


def test_manual_click_shows_terminal_progress_without_hiding_partial_result(app, monkeypatch):
    calls = []

    def update(self, on_event=None):
        calls.append(tuple(self.registry))
        on_event(UpdateEvent("source_started", "shu_main"))
        on_event(UpdateEvent("list_page_read", "shu_main"))
        on_event(UpdateEvent("source_finished", "shu_main"))
        on_event(UpdateEvent("source_started", "undergraduate"))
        on_event(UpdateEvent("source_failed", "undergraduate", "connection_error"))
        return UpdateReport({
            "shu_main": SourceUpdateResult("success", list_pages_read=1),
            "undergraduate": SourceUpdateResult("error", message="connection_error"),
        }, (), state="partial")

    monkeypatch.setattr(UpdateService, "update_all", update)
    app.run()
    assert not calls
    next(item for item in app.sidebar.button if item.label == "更新官网和本科生院").click().run()
    assert not app.exception
    assert calls == [("shu_main", "undergraduate")]
    assert any("部分检查未完成" in item.label for item in app.status)
    assert any(item.value == 100 for item in app.get("progress"))
    assert any("部分检查" in item.value for item in app.warning)
