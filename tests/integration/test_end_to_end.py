import json
from datetime import UTC, datetime
from pathlib import Path

from streamlit.testing.v1 import AppTest

from scripts.export_annotation_queue import export_annotation_queue
from scripts.seed_demo import seed_demo
from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository
from shu_insight.domain.models import (
    DiscoveredNotice,
    NoticeDraft,
    Persona,
    ProofStatus,
    Verification,
)
from shu_insight.ingest.snapshot import import_snapshot
from shu_insight.profiles.service import DEMO_PERSONAS
from shu_insight.services.notices import NoticeQuery, NoticeService
from shu_insight.sources.registry import load_source_registry

AUTHORITY = {
    "shu_main": 1.0,
    "undergraduate": 0.9,
    "graduate": 0.9,
    "computer": 0.95,
}


def test_end_to_end_roles_preserve_verified_only_browse_visibility(
    tmp_path: Path,
) -> None:
    database = seed_demo(tmp_path / "system.sqlite3", reset=True)
    registry = load_source_registry(Path("config/sources.toml"))
    policy = OfficialNoticePolicy({key: adapter.config for key, adapter in registry.items()})
    repository = NoticeRepository(database, policy=policy)
    imported = import_snapshot(Path("data/official/snapshot-2026-09-24.json"), repository)
    service = NoticeService(repository, authority_by_source_id=AUTHORITY,
                            now=lambda: datetime(2026, 9, 25, tzinfo=UTC))
    research_persona, employment_persona = DEMO_PERSONAS[:2]

    research = service.recommend(
        NoticeQuery(terms=("课程",), persona=research_persona)
    )
    employment = service.recommend(
        NoticeQuery(terms=("课程",), persona=employment_persona)
    )

    assert research
    assert employment
    assert imported.imported_count >= 12
    assert len(service.all()) == imported.imported_count
    assert {
        "https://www.shu.edu.cn/info/1051/400395.htm",
        "https://bksy.shu.edu.cn/info/1015/321924.htm",
    } <= {notice.canonical_url for notice in service.all()}
    assert len(repository.list_all()) > len(service.all())
    assert {item.notice.id for item in (*research, *employment)} <= {
        notice.id for notice in service.all()
    }
    assert all("synthetic_demo" not in notice.topic_tags for notice in service.all())
    assert service.detail(research[0].notice.id) == research[0].notice
    assert service.latest()
    assert service.upcoming()


def test_annotation_export_counts_only_audited_official_snapshot(tmp_path: Path) -> None:
    database = seed_demo(tmp_path / "export.sqlite3", reset=True)
    registry = load_source_registry(Path("config/sources.toml"))
    policy = OfficialNoticePolicy({key: adapter.config for key, adapter in registry.items()})
    repository = NoticeRepository(database, policy=policy)
    imported = import_snapshot(Path("data/official/snapshot-2026-09-24.json"), repository)
    output = tmp_path / "queue.jsonl"

    assert imported.imported_count >= 12
    assert export_annotation_queue(repository, output) == imported.imported_count
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == imported.imported_count
    assert {row["original_url"] for row in rows} == {
        notice.canonical_url for notice in repository.list_verified()
    }


def test_personalized_home_preserves_service_tiers_and_listing_evidence(tmp_path):
    database = seed_demo(tmp_path / "personalized-home.sqlite3", reset=True)
    registry = load_source_registry(Path("config/sources.toml"))
    policy = OfficialNoticePolicy({key: adapter.config for key, adapter in registry.items()})
    repository = NoticeRepository(database, policy=policy)
    now = datetime(2026, 10, 6, tzinfo=UTC)
    proof = Verification(ProofStatus.VERIFIED_SNAPSHOT, now, "snapshot:ui-local-fixture")
    for number, title, summary, levels in (
        (210001, "科研申请通知", "请按公开流程办理申请。", ("硕士生",)),
        (210002, "课程安排通知", "学院提供科研交流安排。", ()),
        (210003, "博士课程通知", "请按培养计划办理。", ("博士生",)),
    ):
        repository.upsert(
            NoticeDraft("graduate", f"https://gs.shu.edu.cn/info/1029/{number}.htm",
                        title, summary, education_levels=levels),
            verification=proof,
        )
    repository.upsert_listing(
        DiscoveredNotice("graduate", "https://gs.shu.edu.cn/info/1029/210004.htm", "科研项目通知"),
        list_url="https://gs.shu.edu.cn/xwlb/sy.htm", checked_at=now,
    )
    service = NoticeService(repository, authority_by_source_id=AUTHORITY, now=lambda: now)
    persona = Persona("graduate", "硕士角色", education_level="硕士生")

    def app(service, persona):
        from shu_insight.ui.home import render_home
        from shu_insight.ui.personas import PersonaSelection

        render_home(service, PersonaSelection(persona, ("科研",)))

    at = AppTest.from_function(app, args=(service, persona)).run()
    at = next(r for r in at.radio if r.label == "浏览方式").set_value("为你推荐").run()
    assert not at.exception
    cards = [item.value for item in at.markdown if item.value.startswith("### [")]
    assert len(cards) == 4
    assert "科研申请通知" in cards[0] and "博士课程通知" in cards[-1]
    rendered = " ".join(str(item.value) for kind in ("caption", "markdown", "info")
                        for item in at.get(kind))
    assert all(label in rendered for label in ("直接相关", "可能适用", "其他通知"))
    assert "适用于硕士生" in rendered and "适用于博士生" not in rendered
    assert "仅标题匹配，正文待读取" in rendered and "正文尚未读取，暂无摘要" in rendered
    assert not at.get("progress")
    at = next(r for r in at.radio if r.label == "浏览方式").set_value("全部通知").run()
    for order in ("最新发布", "即将截止"):
        at = next(r for r in at.radio if r.label == "通知排序").set_value(order).run()
        assert not at.exception
        visible = {item.value for item in at.markdown if item.value.startswith("### [")}
        assert visible == set(cards)
