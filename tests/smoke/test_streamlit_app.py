import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from shutil import copyfile

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from scripts.seed_demo import seed_demo
from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import (
    ArticleReadRepository,
    NoticeRepository,
    PersonaRepository,
    SearchHistoryRepository,
)
from shu_insight.db.schema import Database
from shu_insight.domain.models import (
    DeadlineConfidence,
    DiscoveredNotice,
    NoticeDraft,
    Persona,
    ProofStatus,
    Verification,
)
from shu_insight.ingest.http_client import PoliteHttpClient
from shu_insight.ingest.pipeline import IngestionPipeline
from shu_insight.ingest.preload import import_preload
from shu_insight.ingest.snapshot import import_snapshot
from shu_insight.profiles.learning import LEARN_LABEL
from shu_insight.profiles.service import DEMO_PERSONAS
from shu_insight.services.notices import NoticeService
from shu_insight.services.update import UpdateService
from shu_insight.settings import AppSettings
from shu_insight.sources.registry import load_source_registry

APP_PATH = Path(__file__).resolve().parents[2] / "app.py"


def rendered_status(at):
    return " ".join(str(x.value) for kind in
                    ("caption", "markdown", "success", "warning", "error", "info")
                    for x in at.get(kind))


def card_titles(at):
    headings = [w.value for w in at.markdown if w.value.startswith("### [")]
    links = [w.proto.label for w in at.get("link_button")
             if "notice-title-" in getattr(w.proto, "id", "")]
    return headings + links


@pytest.fixture
def reviewed_service_day(monkeypatch):
    """Service guard stays deterministic; production still uses date.today()."""
    original_init = UpdateService.__init__
    def initialize(self, **kwargs):
        original_init(self, **kwargs, today=lambda: date(2026, 10, 7))

    monkeypatch.setattr(UpdateService, "__init__", initialize)


@pytest.fixture
def local_http(monkeypatch, reviewed_service_day):
    """Real polite client and service, with only the HTTP boundary replaced."""
    clients, requests, events = [], [], []
    original_init = PoliteHttpClient.__init__
    original_run = IngestionPipeline.run

    def handler(request):
        requests.append(request)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        title = "拟录取名单公示"
        href = "/info/1/1.htm"
        listings = {
            "www.shu.edu.cn": (
                '<div class="ej_main"><div class="w1400"><div class="list"><ul><li>'
                f'<a href="{href}"><p class="bt">{title}</p><p class="sj">2026-10-07</p></a>'
                '</li></ul></div></div></div>'
            ),
            "bksy.shu.edu.cn": (
                '<div id="body"><div class="only-listPage"><div class="only-list"><ul><li>'
                f'<a href="{href}">{title}</a><span>2026-10-07</span>'
                '</li></ul></div></div></div>'
            ),
            "gs.shu.edu.cn": (
                '<table width="755"><tr id="line_u17_0"><td>'
                f'<a href="{href}">{title}</a></td><td>2026/10/07</td></tr></table>'
            ),
            "cs.shu.edu.cn": (
                '<div class="tzgg"><ul><li>'
                f'<a href="{href}"><div class="tz-tx"><h3>{title}</h3></div>'
                '<div class="tz-d"><span>2026-10</span><b>07</b></div></a>'
                '</li></ul></div>'
            ),
        }
        return httpx.Response(200, text=listings[request.url.host],
                              headers={"Content-Type": "text/html"})

    def initialize(self, **kwargs):
        clients.append(self)
        original_init(self, **kwargs, transport=httpx.MockTransport(handler), sleep=lambda _: None)

    def run(self, *args, **kwargs):
        report = original_run(self, *args, **kwargs)
        events.extend(report.events)
        return report

    monkeypatch.setattr(PoliteHttpClient, "__init__", initialize)
    monkeypatch.setattr("shu_insight.ingest.pipeline.IngestionPipeline.run", run)
    return clients, requests, events


@pytest.fixture
def app_test(tmp_path: Path, monkeypatch) -> AppTest:
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(tmp_path / "app.sqlite3"))
    monkeypatch.delenv("SHU_NOTICE_CONTACT", raising=False)
    monkeypatch.delenv("SHU_NOTICE_AI_ENABLED", raising=False)
    return AppTest.from_file(APP_PATH, default_timeout=15)


def test_app_shows_non_official_notice_and_two_browse_modes(app_test: AppTest) -> None:
    at = app_test.run()

    assert not at.exception
    assert any("非上海大学官方产品" in markdown.value for markdown in at.markdown)
    assert len(at.tabs) == 0
    assert next(r for r in at.radio if r.label == "浏览方式").options == [
        "全部通知", "为你推荐"
    ]
    assert next(r for r in at.radio if r.label == "浏览方式").value == "为你推荐"


def test_all_notices_offers_latest_and_upcoming_sort(app_test: AppTest) -> None:
    at = app_test.run()
    at = next(r for r in at.radio if r.label == "浏览方式").set_value("全部通知").run()

    assert not at.exception
    assert next(r for r in at.radio if r.label == "通知排序").options == [
        "最新发布", "即将截止"
    ]


def test_first_open_shows_verified_notices_without_profile_or_keywords(
    tmp_path: Path, monkeypatch,
) -> None:
    path = tmp_path / "first-open.sqlite3"
    database = seed_demo(path, reset=True)
    registry = load_source_registry(Path("config/sources.toml"))
    policy = OfficialNoticePolicy({key: adapter.config for key, adapter in registry.items()})
    imported = import_snapshot(Path("data/official/snapshot-2026-09-24.json"),
                               NoticeRepository(database, policy=policy))
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(path))

    at = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert not at.exception
    at = next(r for r in at.radio if r.label == "浏览方式").set_value("全部通知").run()
    assert next(r for r in at.radio if r.label == "浏览方式").value == "全部通知"
    cards = card_titles(at)
    assert len(cards) == imported.imported_count
    snapshot = json.loads(Path("data/official/snapshot-2026-09-24.json").read_text(
        encoding="utf-8"
    ))["entries"]
    latest_date = max(entry["published_at"] for entry in snapshot if entry["published_at"])
    latest_urls = {
        entry["canonical_url"] for entry in snapshot if entry["published_at"] == latest_date
    }
    assert any(url in cards[0] for url in latest_urls)
    assert any("https://www.shu.edu.cn/info/1051/400395.htm" in card for card in cards)
    assert any("https://bksy.shu.edu.cn/info/1015/321924.htm" in card for card in cards)


def test_deadline_sort_in_all_notices_keeps_historical_notices(
    tmp_path: Path, monkeypatch,
) -> None:
    path = tmp_path / "deadline-sort.sqlite3"
    database = seed_demo(path, reset=True)
    registry = load_source_registry(Path("config/sources.toml"))
    policy = OfficialNoticePolicy({key: adapter.config for key, adapter in registry.items()})
    imported = import_snapshot(Path("data/official/snapshot-2026-09-24.json"),
                               NoticeRepository(database, policy=policy))
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(path))

    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at = next(r for r in at.radio if r.label == "浏览方式").set_value("全部通知").run()
    at = next(r for r in at.radio if r.label == "通知排序").set_value("即将截止").run()

    assert not at.exception
    cards = card_titles(at)
    assert len(cards) == imported.imported_count
    assert any("https://www.shu.edu.cn/info/1051/400245.htm" in card for card in cards)
    assert any("已过截止" in w.value for w in at.subheader)


def test_recommend_without_terms_shows_ranked_verified_cards(
    tmp_path: Path, monkeypatch, local_http,
) -> None:
    path = tmp_path / "recommend-empty-terms.sqlite3"
    database = seed_demo(path, reset=True)
    registry = load_source_registry(Path("config/sources.toml"))
    policy = OfficialNoticePolicy({key: adapter.config for key, adapter in registry.items()})
    repository = NoticeRepository(database, policy=policy)
    imported = import_snapshot(Path("data/official/snapshot-2026-09-24.json"), repository)
    # More than a typical first-screen cap; every list and parsed card must remain reachable.
    for index in range(24):
        repository.upsert_listing(
            DiscoveredNotice("graduate", f"https://gs.shu.edu.cn/info/1029/{200000 + index}.htm",
                             f"课程公开通知{index}"),
            list_url="https://gs.shu.edu.cn/xwlb/sy.htm",
            checked_at=datetime(2026, 10, 6, tzinfo=UTC),
        )
    repository.upsert(
        NoticeDraft("graduate", "https://gs.shu.edu.cn/info/1029/210099.htm",
                    "硕士课程办理通知", "请按公开流程办理课程。", education_levels=("硕士生",)),
        verification=Verification(ProofStatus.VERIFIED_SNAPSHOT,
                                  datetime(2026, 10, 6, tzinfo=UTC), "snapshot:ui-local-fixture"),
    )
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(path))

    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at = next(r for r in at.radio if r.label == "浏览方式").set_value("为你推荐").run()

    assert not at.exception
    cards = card_titles(at)
    assert len(cards) == imported.imported_count + 25
    assert "其他通知" in rendered_status(at) and "暂无明确匹配证据" in rendered_status(at)
    assert "尚未设置推荐条件" not in rendered_status(at)
    assert "仅标题匹配" not in rendered_status(at)
    assert not next(w for w in at.multiselect if w.label == "兴趣标签（最多 8 个）").value
    assert next(w for w in at.text_input if w.label == "关键词（最多 5 个）").value == ""
    # Switching between both timeline orders cannot hide real cards or start updates.
    at = next(r for r in at.radio if r.label == "浏览方式").set_value("全部通知").run()
    for order in ("最新发布", "即将截止"):
        at = next(r for r in at.radio if r.label == "通知排序").set_value(order).run()
        assert not at.exception
        assert set(card_titles(at)) == set(cards)
    assert not local_http[0] and not local_http[1]


def test_browse_controls_are_not_duplicated(app_test, local_http):
    at = app_test.run()
    for mode in ("为你推荐", "全部通知", "为你推荐", "全部通知"):
        assert len([r for r in at.radio if r.label == "浏览方式"]) == 1
        at = next(r for r in at.radio if r.label == "浏览方式").set_value(mode).run()
        assert not at.exception and not at.tabs
        assert len([r for r in at.radio if r.label == "浏览方式"]) == 1
        assert len([r for r in at.radio if r.label == "通知排序"]) == (mode == "全部通知")
    assert not local_http[0] and not local_http[1]


def test_demo_only_database_is_empty_in_every_browse_mode(tmp_path, monkeypatch):
    path = tmp_path / "demo.sqlite3"
    seed_demo(path, reset=True)
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(path))
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception and len(at.tabs) == 0
    for mode in ("为你推荐", "全部通知"):
        at = next(r for r in at.radio if r.label == "浏览方式").set_value(mode).run()
        rendered = " ".join(str(x.value) for kind in ("markdown", "caption", "info")
                            for x in at.get(kind))
        assert not at.exception
        assert "[演示]" not in rendered and "demo.invalid" not in rendered
        assert "尚无已核实的官网通知" in rendered


def test_verified_card_shows_full_summary_official_link_and_unknown_dates(tmp_path, monkeypatch):
    path = tmp_path / "real.sqlite3"
    database = Database(path)
    database.initialize()
    registry = load_source_registry(Path("config/sources.toml"))
    policy = OfficialNoticePolicy({key: adapter.config for key, adapter in registry.items()})
    manifest = json.loads(Path("data/official/snapshot-2026-09-24.json").read_text(
        encoding="utf-8"))
    entry = next(row for row in manifest["entries"] if row["id"] == "gs-173242")
    summary = entry["summary"]
    url = entry["canonical_url"]
    title = entry["title"]
    repository = NoticeRepository(database, policy=policy)
    repository.upsert(NoticeDraft(
        source_id="graduate", canonical_url=url,
        title="UNVERIFIED_LEGACY_TITLE", normalized_text="UNVERIFIED_LEGACY_SUMMARY",
    ))
    repository.upsert(
        # Omit known dates here to exercise the UI's missing-metadata state.
        NoticeDraft(source_id="graduate", canonical_url=url, title=title,
                    normalized_text=summary),
        verification=Verification(ProofStatus.VERIFIED_SNAPSHOT,
                                  datetime.fromisoformat(entry["verified_at"]),
                                  "snapshot:gs-173242"),
        summary_override=summary,
    )
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(path))
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    at = next(r for r in at.radio if r.label == "浏览方式").set_value("全部通知").run()
    rendered = " ".join(str(x.value) for x in at.markdown)
    captions = " ".join(x.value for x in at.caption)
    assert not at.exception
    assert summary in rendered
    assert f"[{title}]({url})" in rendered
    assert "发布：未知" in captions and "截止：未知" in captions
    assert "官网快照" in captions
    assert at.dataframe
    visible_detail = " ".join(
        json.dumps(item.value.to_dict(orient="records"), ensure_ascii=False)
        for item in at.dataframe
    )
    assert "UNVERIFIED_LEGACY_TITLE" not in visible_detail
    assert "UNVERIFIED_LEGACY_SUMMARY" not in visible_detail


def test_expired_verified_notice_has_historical_label_not_empty_database_message(
    tmp_path, monkeypatch,
):
    path = tmp_path / "historical.sqlite3"
    database = Database(path)
    database.initialize()
    registry = load_source_registry(Path("config/sources.toml"))
    policy = OfficialNoticePolicy({key: adapter.config for key, adapter in registry.items()})
    manifest = json.loads(Path("data/official/snapshot-2026-09-24.json").read_text(
        encoding="utf-8"))
    entry = next(row for row in manifest["entries"] if row["id"] == "shu-main-400245")
    NoticeRepository(database, policy=policy).upsert(
        NoticeDraft(source_id=entry["source_id"], canonical_url=entry["canonical_url"],
                    title=entry["title"], normalized_text=entry["summary"],
                    published_at=datetime.fromisoformat(entry["published_at"] + "T00:00:00+08:00"),
                    deadline_at=datetime.fromisoformat(entry["deadline_at"]),
                    deadline_confidence=DeadlineConfidence.HIGH),
        verification=Verification(ProofStatus.VERIFIED_SNAPSHOT,
                                  datetime.fromisoformat(entry["verified_at"]),
                                  "snapshot:shu-main-400245"),
        summary_override=entry["summary"],
    )
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(path))
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    at = next(r for r in at.radio if r.label == "浏览方式").set_value("全部通知").run()
    assert any("历史通知" in caption.value for caption in at.caption)
    at = next(r for r in at.radio if r.label == "通知排序").set_value("即将截止").run()
    assert not any("尚无已核实的官网通知" in info.value for info in at.info)


def test_audited_date_only_deadline_shows_date_without_fabricated_clock(tmp_path, monkeypatch):
    path = tmp_path / "snapshot.sqlite3"
    database = seed_demo(path, reset=True)
    registry = load_source_registry(Path("config/sources.toml"))
    policy = OfficialNoticePolicy({key: adapter.config for key, adapter in registry.items()})
    import_snapshot(Path("data/official/snapshot-2026-09-24.json"),
                    NoticeRepository(database, policy=policy))
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(path))
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    for mode in ("为你推荐", "全部通知"):
        at = next(r for r in at.radio if r.label == "浏览方式").set_value(mode).run()
        rendered = " ".join(str(x.value) for kind in ("markdown", "caption", "info")
                            for x in at.get(kind))
        assert "demo.invalid" not in rendered and "[演示]" not in rendered
    captions = " ".join(x.value for x in at.caption)
    assert "截止：2026-10-11（待核实）" in captions
    assert "2026-10-11 00:00" not in captions
    assert "发布：2026-09-11" in captions
    assert "2026-09-11 00:00" not in captions


def test_app_recommend_uses_ranking_config(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "ranked.sqlite3"
    database = Database(database_path)
    database.initialize()
    registry = load_source_registry(Path("config/sources.toml"))
    policy = OfficialNoticePolicy({key: adapter.config for key, adapter in registry.items()})
    import_snapshot(
        Path("data/official/snapshot-2026-09-24.json"),
        NoticeRepository(database, policy=policy),
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for filename in ("sources.toml", "taxonomy.toml"):
        copyfile(Path("config") / filename, config_dir / filename)
    with (config_dir / "taxonomy.toml").open("a", encoding="utf-8") as taxonomy:
        taxonomy.write('\n[tags."免修"]\nkeywords = ["免修"]\n')
    (config_dir / "ranking.toml").write_text(
        "[profile]\n"
        "relevance = 0.40\n"
        "audience = 0.25\n"
        "urgency = 0.20\n"
        "authority = 0.10\n"
        "freshness = 0.05\n"
        "[no_profile]\n"
        "relevance = 0.0\n"
        "urgency = 0.0\n"
        "authority = 1.0\n"
        "freshness = 0.0\n",
        encoding="utf-8",
    )
    original_load = AppSettings.load
    monkeypatch.setattr(
        AppSettings,
        "load",
        classmethod(lambda cls, root: replace(original_load(root), config_dir=config_dir)),
    )
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(database_path))
    monkeypatch.delenv("SHU_NOTICE_CONTACT", raising=False)
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at = next(w for w in at.multiselect if w.label == "兴趣标签（最多 8 个）").set_value(
        ["免修"]).run()
    at = next(w for w in at.radio if w.label == "浏览方式").set_value(
        "为你推荐"
    ).run()

    assert not at.exception
    cards = card_titles(at)
    # The only keyword hit remains first by tier; authority orders the remaining cards.
    assert "https://gs.shu.edu.cn/info/1029/173242.htm" in cards[0]
    assert "https://www.shu.edu.cn/" in cards[1]
    assert len(cards) == 12
    assert not at.get("progress") and "综合相关度" not in rendered_status(at)


def widget(at, kind, label):
    return next(w for w in getattr(at, kind) if w.label == label)


def test_first_open_offers_empty_interest_search_and_five_sidebar_groups(
    app_test, tmp_path, local_http,
):
    at = app_test.run()
    assert not at.exception
    assert widget(at, "multiselect", "兴趣标签（最多 8 个）").value == []
    assert widget(at, "text_input", "关键词（最多 5 个）").value == ""
    assert [h.value for h in at.sidebar.header] == [
        "兴趣标签", "关键词搜索", "我的画像", "更新通知", "收录统计"]
    persona_picker = widget(at, "selectbox", "按画像排序（可选）")
    assert persona_picker.value == 0
    assert persona_picker.options[0] == "不使用画像（通用排序）"
    assert persona_picker.options[1].startswith(LEARN_LABEL)
    assert list(persona_picker.options[2:]) == [p.label for p in DEMO_PERSONAS]
    assert len(persona_picker.options) == 2 + len(DEMO_PERSONAS)
    assert len(at.sidebar.button) == 1
    assert at.sidebar.button[0].label == "更新官网和本科生院"
    assert not at.sidebar.button[0].disabled
    forbidden = {"页面", "联系邮箱（可选）", "确认删除全部本地角色", "删除全部本地角色",
                 "清空未核实候选队列（只删除队列）", "复制当前角色", "重置当前角色",
                 "当前角色", "培养层次", "年级", "学院", "专业", "校区", "发展目标",
                 "角色名称", "保存当前角色", "保存为本地角色"}
    assert not forbidden.intersection(w.label for kind in ("button", "text_input", "selectbox",
                                                           "checkbox") for w in at.get(kind))
    assert not at.sidebar.expander
    assert PersonaRepository(Database(tmp_path / "app.sqlite3")).list_all() == []
    assert not local_http[0] and not local_http[1]


def test_interest_search_ignores_legacy_role_state_and_keeps_existing_profiles(
    app_test, tmp_path,
):
    database = Database(tmp_path / "app.sqlite3")
    database.initialize()
    persona = Persona("saved-persona", "已有画像", education_level="硕士生",
                      interest_tags=("科研",), custom_keywords=("Python",))
    PersonaRepository(database).save(persona)
    app_test.session_state["active-persona"] = "本地 · 已有画像 [saved-persona]"
    app_test.session_state["selected-persona-id"] = persona.id
    app_test.session_state["persona-tags-widget"] = ["科研"]
    app_test.session_state["persona-keywords-widget"] = "Python"
    at = app_test.run()
    assert not at.exception
    assert widget(at, "multiselect", "兴趣标签（最多 8 个）").value == []
    assert widget(at, "text_input", "关键词（最多 5 个）").value == ""
    assert widget(at, "selectbox", "按画像排序（可选）").value == 0
    at = widget(at, "multiselect", "兴趣标签（最多 8 个）").set_value(["竞赛"]).run()
    at = widget(at, "text_input", "关键词（最多 5 个）").set_value("课程").run()
    at = at.run()
    assert widget(at, "multiselect", "兴趣标签（最多 8 个）").value == ["竞赛"]
    assert widget(at, "text_input", "关键词（最多 5 个）").value == "课程"
    assert PersonaRepository(database).list_all() == [persona]


def test_interest_options_include_current_taxonomy_and_limit_selection(app_test):
    at = app_test.run()
    assert not at.exception
    tags = widget(at, "multiselect", "兴趣标签（最多 8 个）")
    assert {"保研", "教务", "奖助评优", "讲座活动"} <= set(tags.options)
    assert tags.proto.max_selections == 8


def test_persona_mode_explains_audience_and_score_breakdown(tmp_path):
    database = seed_demo(tmp_path / "app.sqlite3", reset=True)
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    baseline = rendered_status(at)
    assert "画像模式" not in baseline

    at = widget(at, "selectbox", "按画像排序（可选）").set_value(2).run()

    assert not at.exception
    assert widget(at, "selectbox", "按画像排序（可选）").value == 2
    rendered = rendered_status(at)
    assert "画像模式" in rendered
    assert "受众契合占 25%" in rendered
    assert "为什么排在这里？" in [expander.label for expander in at.expander]
    assert "受众契合" in rendered
    assert "适用于" in rendered
    assert {persona.id for persona in PersonaRepository(database).list_all()} == {
        persona.id for persona in DEMO_PERSONAS
    }


def test_learned_persona_explains_missing_history(app_test):
    at = app_test.run()
    picker = widget(at, "selectbox", "按画像排序（可选）")
    assert picker.options[1].startswith(LEARN_LABEL)
    assert picker.options[1] != LEARN_LABEL
    at = picker.set_value(1).run()
    assert not at.exception
    rendered = rendered_status(at)
    assert "聚类画像暂不可用" in rendered
    assert "10 篇不同文章" in rendered
    assert "画像模式" not in rendered
    assert widget(at, "selectbox", "按画像排序（可选）").value == 0


def test_learned_persona_clusters_distinct_opened_articles(tmp_path, monkeypatch):
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(tmp_path / "app.sqlite3"))
    database = Database(tmp_path / "app.sqlite3")
    database.initialize()
    sources = load_source_registry(Path("config/sources.toml"))
    repository = NoticeRepository(database, policy=OfficialNoticePolicy(
        {key: adapter.config for key, adapter in sources.items()}))
    import_preload(Path("data/official/preload-2026-10-25.json"), repository)
    readings = ArticleReadRepository(database)
    for notice in repository.list_browseable()[:10]:
        readings.record(notice.id)
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    picker = widget(at, "selectbox", "按画像排序（可选）")
    assert picker.options[1] == LEARN_LABEL
    at = picker.set_value(1).run()

    assert not at.exception
    rendered = rendered_status(at)
    assert "聚类画像：" in rendered
    assert "我的阅读兴趣·" in rendered
    assert "画像模式" in rendered
    assert "已用聚类画像的目标与代表词" in rendered
    assert "为什么排在这里？" in [expander.label for expander in at.expander]
    assert readings.count() == 10
    assert SearchHistoryRepository(database).count() == 0


def test_only_article_link_clicks_count_and_title_footer_deduplicate(tmp_path, monkeypatch):
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(tmp_path / "clicks.sqlite3"))
    database = Database(tmp_path / "clicks.sqlite3")
    database.initialize()
    sources = load_source_registry(Path("config/sources.toml"))
    repository = NoticeRepository(database, policy=OfficialNoticePolicy(
        {key: adapter.config for key, adapter in sources.items()}))
    import_preload(Path("data/official/preload-2026-10-25.json"), repository)
    import streamlit as st

    original_link_button = st.link_button
    callbacks = {}

    def link_button(label, url, **kwargs):
        if callable(kwargs.get("on_click")):
            callbacks[kwargs["key"]] = (kwargs["on_click"], kwargs["args"], url)
        return original_link_button(label, url, **kwargs)

    monkeypatch.setattr(st, "link_button", link_button)
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at = widget(at, "multiselect", "兴趣标签（最多 8 个）").set_value(["科研"]).run()
    at = widget(at, "selectbox", "按画像排序（可选）").set_value(2).run()
    assert not at.exception
    readings = ArticleReadRepository(database)
    assert readings.count() == 0
    assert SearchHistoryRepository(database).count() == 0
    key = next(key for key in callbacks if key.startswith("notice-title-"))
    callback, args, url = callbacks[key]
    assert url == args[0].canonical_url
    callback(*args)
    at = at.run()
    assert readings.count() == 1
    footer = key.replace("notice-title-", "notice-original-", 1)
    callback, args, _ = callbacks[footer]
    callback(*args)
    at = at.run()
    assert readings.count() == 1
    assert "已打开 1 篇，还需 9 篇" in rendered_status(at)
    at = widget(at, "radio", "浏览方式").set_value("全部通知").run()
    assert not at.exception and readings.count() == 1


def test_recommendation_search_filters_literal_title_evidence_before_rendering(tmp_path):
    database = Database(tmp_path / "search-home.sqlite3")
    database.initialize()
    sources = load_source_registry(Path("config/sources.toml"))
    repository = NoticeRepository(database, policy=OfficialNoticePolicy(
        {key: adapter.config for key, adapter in sources.items()}))
    for number, title in ((400980, "Python 课程通知"), (400981, "图书馆开放通知")):
        repository.upsert_listing(
            DiscoveredNotice("shu_main", f"https://www.shu.edu.cn/info/1051/{number}.htm",
                             title),
            list_url=sources["shu_main"].config.list_url,
            checked_at=datetime(2026, 10, 7, tzinfo=UTC),
        )
    with database.connect() as conn:
        conn.execute("UPDATE notices SET summary = 'Python 旧正文'")
    service = NoticeService(repository, authority_by_source_id={"shu_main": 1.0})

    def app(service):
        from shu_insight.ui.home import render_home
        from shu_insight.ui.personas import PersonaSelection

        render_home(service, PersonaSelection(None, ("python",), ("python",)))

    at = AppTest.from_function(app, args=(service,)).run()

    assert not at.exception
    cards = card_titles(at)
    assert len(cards) == 1 and "Python 课程通知" in cards[0]
    assert "命中 1 条通知" in rendered_status(at)


@pytest.mark.parametrize("mode", ["为你推荐", "全部通知"])
def test_keywords_filter_allowed_evidence_immediately_and_clear(
    app_test, tmp_path, local_http, mode,
):
    db = Database(tmp_path / "app.sqlite3")
    db.initialize()
    sources = load_source_registry(Path("config/sources.toml"))
    repo = NoticeRepository(db, policy=OfficialNoticePolicy(
        {key: adapter.config for key, adapter in sources.items()}))
    repo.upsert(NoticeDraft("shu_main", "https://www.shu.edu.cn/info/1051/400997.htm",
                            "课程申请通知", "Python 学习课程安排，请按流程办理课程。"),
                verification=Verification(ProofStatus.VERIFIED_SNAPSHOT,
                                          datetime(2026, 10, 7, tzinfo=UTC), "snapshot:search"))
    repo.upsert_listing(DiscoveredNotice("graduate", "https://gs.shu.edu.cn/info/1029/200001.htm",
                                         "研究生公开讲座通知"),
                        list_url="https://gs.shu.edu.cn/xwlb/sy.htm",
                        checked_at=datetime(2026, 10, 7, tzinfo=UTC))
    # Legacy body must never turn list-only title evidence into a keyword hit.
    with db.connect() as conn:
        conn.execute("UPDATE notices SET summary = 'Python 旧正文' WHERE source_id = 'graduate'")
    at = app_test.run()
    at = widget(at, "radio", "浏览方式").set_value(mode).run()
    assert len(card_titles(at)) == 2
    assert "当前显示 2 条通知" in rendered_status(at)
    at = widget(at, "text_input", "关键词（最多 5 个）").set_value("python").run()
    cards = card_titles(at)
    assert len(cards) == 1 and "课程申请通知" in cards[0]
    assert "命中 1 条通知" in rendered_status(at)
    at = widget(at, "text_input", "关键词（最多 5 个）").set_value(
        "无一,无二，无三,无四，无五,python").run()
    assert not card_titles(at)
    assert "仅使用前 5 个关键词" in rendered_status(at)
    at = widget(at, "text_input", "关键词（最多 5 个）").set_value(
        "无一,无二，无三,无四，python").run()
    assert len(card_titles(at)) == 1
    if mode == "全部通知":
        at = widget(at, "radio", "通知排序").set_value("即将截止").run()
        assert len(card_titles(at)) == 1
    at = widget(at, "text_input", "关键词（最多 5 个）").set_value(
        "PYTHON,python，无一,无二,无三，讲座").run()
    assert len(card_titles(at)) == 2
    assert "命中 2 条通知" in rendered_status(at)
    at = widget(at, "text_input", "关键词（最多 5 个）").set_value("不存在的关键词").run()
    assert not card_titles(at)
    assert "没有匹配关键词的通知" in rendered_status(at)
    assert "命中 0 条通知" in rendered_status(at)
    at = widget(at, "button", "清空关键词").click().run()
    assert widget(at, "text_input", "关键词（最多 5 个）").value == ""
    assert len(card_titles(at)) == 2
    assert not at.exception and not local_http[0] and not local_http[1]


def test_manual_update_with_full_registry_only_contacts_two_sources(app_test, local_http, tmp_path):
    at = app_test.run().run()
    assert not local_http[0] and not local_http[1]
    at = widget(at, "button", "更新官网和本科生院").click().run()
    assert not at.exception
    assert {r.url.host for r in local_http[1]} == {"www.shu.edu.cn", "bksy.shu.edu.cn"}
    assert set(at.session_state["manual-update-report"].by_source) == {"shu_main", "undergraduate"}
    assert "2 个来源均未完成检查" in rendered_status(at)
    assert "四个来源均完成" not in rendered_status(at)
    assert not at.sidebar.success and not at.sidebar.warning and not at.sidebar.error
    before = len(local_http[1])
    at = at.run()
    assert len(local_http[1]) == before
    with Database(tmp_path / "app.sqlite3").connect() as conn:
        assert {r[0] for r in conn.execute("SELECT source_id FROM source_status")} == {
            "shu_main", "undergraduate"}


def test_interests_expand_synonyms_but_keyword_search_stays_literal_in_both_modes(
    app_test, tmp_path, local_http,
):
    db = Database(tmp_path / "app.sqlite3")
    db.initialize()
    sources = load_source_registry(Path("config/sources.toml"))
    repo = NoticeRepository(db, policy=OfficialNoticePolicy(
        {key: adapter.config for key, adapter in sources.items()}))
    titles = ("推免报名通知", "夏令营申请通知", "学术交流通知", "论文讲座通知", "图书馆开放通知")
    for index, title in enumerate(titles):
        repo.upsert_listing(
            DiscoveredNotice("shu_main", f"https://www.shu.edu.cn/info/1051/{400990 + index}.htm",
                             title),
            list_url=sources["shu_main"].config.list_url,
            checked_at=datetime(2026, 10, 7, tzinfo=UTC),
        )
    at = app_test.run()
    assert not at.exception
    assert "匹配关键词" not in rendered_status(at)
    at = widget(at, "multiselect", "兴趣标签（最多 8 个）").set_value(["保研", "科研"]).run()
    text = rendered_status(at)
    for term in ("推免", "夏令营", "学术", "论文"):
        assert f"匹配关键词：{term}" in text
    assert sum("仅标题匹配，正文待读取" in caption.value for caption in at.caption) == 4
    cards = card_titles(at)
    assert len(cards) == 5 and "图书馆开放通知" in cards[-1]
    at = widget(at, "text_input", "关键词（最多 5 个）").set_value("保研").run()
    for mode in ("为你推荐", "全部通知"):
        at = widget(at, "radio", "浏览方式").set_value(mode).run()
        assert not card_titles(at)
        assert "没有匹配关键词的通知" in rendered_status(at)
    at = widget(at, "text_input", "关键词（最多 5 个）").set_value("推免").run()
    cards = card_titles(at)
    assert len(cards) == 1 and "推免报名通知" in cards[0]
    assert not at.exception and not local_http[0] and not local_http[1]
