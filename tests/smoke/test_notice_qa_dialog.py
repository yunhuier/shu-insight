from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository
from shu_insight.db.schema import Database
from shu_insight.ingest.preload import import_preload
from shu_insight.sources.registry import load_source_registry

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def app(tmp_path, monkeypatch):
    from shu_insight.services.notice_fulltext import FulltextResult

    # UI-only runs never visit school sites. The actual fulltext reader is
    # independently exercised with MockTransport in test_notice_fulltext.py.
    monkeypatch.setattr("shu_insight.ui.notice_qa.prepare_fulltext",
                        lambda evidence, question, cache, **kwargs:
                        FulltextResult(tuple(evidence), (), 0, 0))
    database = Database(tmp_path / "qa.sqlite3")
    database.initialize()
    registry = load_source_registry(ROOT / "config/sources.toml")
    import_preload(ROOT / "data/official/preload-2026-10-25.json", NoticeRepository(
        database, OfficialNoticePolicy({key: item.config for key, item in registry.items()})))
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(database.path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)


def test_each_card_has_dialog_button_and_opening_requires_no_provider_call(app):
    app.run()
    buttons = [item for item in app.button if str(item.key).startswith("notice-qa-")]
    assert len(buttons) == 74
    buttons[0].click().run()
    assert not app.exception
    assert any(item.label == "问题" for item in app.text_area)
    assert next(item for item in app.button if item.label == "发送问题").disabled


def test_explicit_submit_calls_once_and_another_notice_has_separate_chat(app, monkeypatch):
    from shu_insight.services.notice_qa import DeepSeekClient, QaAnswer

    calls = []

    def answer(self, question, evidence, history):
        calls.append((question, evidence, history))
        return QaAnswer("已收录资料没有给出完整材料清单。", (evidence[0],), "test-model")

    monkeypatch.setattr(DeepSeekClient, "answer", answer)
    app.run()
    assert not calls
    buttons = [item for item in app.button if str(item.key).startswith("notice-qa-")]
    second = buttons[1].key
    first = buttons[0].key
    buttons[0].click().run()
    assert not calls
    # AppTest reruns the whole script, whereas browser dialog widgets rerun
    # their fragment. Reopen the same dialog explicitly for that harness run.
    app.text_input(key="deepseek-api-key").set_value("test-key-not-real")
    app.button(key=first).click().run()
    assert not calls
    next(item for item in app.text_area if item.label == "问题").set_value("需要什么材料？")
    app.button(key=first).click()
    next(item for item in app.button if item.label == "发送问题").click().run()
    assert not app.exception and len(calls) == 1
    assert calls[0][0] == "需要什么材料？" and not calls[0][2]
    assert len(app.chat_message) == 2
    app.button(key=second).click().run()
    assert not app.exception and not app.chat_message and len(calls) == 1


def test_long_history_does_not_displace_the_latest_question_from_retrieval(app, monkeypatch):
    from shu_insight.services.notice_qa import DeepSeekClient, QaAnswer
    from shu_insight.ui import notice_qa

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    seen = []
    original = notice_qa.retrieve_evidence

    def retrieve(records, anchor, question, **kwargs):
        seen.append(question[:1000])
        return original(records, anchor, question, **kwargs)

    monkeypatch.setattr(notice_qa, "retrieve_evidence", retrieve)
    monkeypatch.setattr(DeepSeekClient, "answer", lambda self, question, evidence, history:
                        QaAnswer("本地摘要未说明。", (evidence[0],), "test-model"))
    app.run()
    first = next(item for item in app.button if str(item.key).startswith("notice-qa-"))
    key = first.key
    first.click().run()
    current = next(iter(app.session_state["notice-qa-conversations"].values()))
    current["turns"] = [{"question": "旧话题" * 400, "answer": "旧答复", "sources": ()}]
    next(item for item in app.text_area if item.label == "问题").set_value("最新问题奖学金")
    app.button(key=key).click()
    next(item for item in app.button if item.label == "发送问题").click().run()
    assert not app.exception
    assert len(seen) == 1 and "最新问题奖学金" in seen[0]
