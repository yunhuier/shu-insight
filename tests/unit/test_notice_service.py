from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository
from shu_insight.db.schema import Database
from shu_insight.domain.models import (
    DeadlineConfidence,
    DiscoveredNotice,
    NoticeDraft,
    Persona,
    ProofStatus,
    Verification,
)
from shu_insight.ranking.score import load_ranking_weights
from shu_insight.services.notices import NoticeQuery, NoticeService
from shu_insight.sources.registry import load_source_registry

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
NO_MATCH_QUERY = NoticeQuery(terms=("量子航天不存在",))
HISTORICAL_PERSON_TABLE = (
    "序号 姓名 原学士学位证书编号 "
    "1 样例甲 1234567890123456 2 样例乙 2345678901234567"
)


@pytest.fixture
def service(tmp_path: Path) -> NoticeService:
    database = Database(tmp_path / "notices.sqlite3")
    database.initialize()
    registry = load_source_registry(Path("config/sources.toml"))
    policy = OfficialNoticePolicy({key: item.config for key, item in registry.items()})
    repository = NoticeRepository(database, policy=policy)
    # Main-site dates have no page clock time; midnight UTC is a test sorting anchor.
    drafts = (
        NoticeDraft(
            source_id="graduate",
            canonical_url="https://gs.shu.edu.cn/info/1029/173242.htm",
            title="上海大学2026级硕博连读研究生、直博生 申请课程免修及成绩认定通知",
            normalized_text=(
                "2026级硕博连读研究生和直博生可按通知条件申请课程免修或已修课程学分、成绩认定。"
                "申请时间为2026年9月11日至10月11日，需完成个人培养计划后在研究生综合服务平台申请，"
                "依次经导师、学院和研究生院审核。"
            ),
            published_at=datetime(2026, 9, 11, 5, 50, tzinfo=UTC),
            # Date-only deadline: UTC midnight is a sorting anchor, not a claimed hour.
            deadline_at=datetime(2026, 10, 11, tzinfo=UTC),
            deadline_confidence=DeadlineConfidence.LOW,
            education_levels=("研究生",),
            categories=("培养管理",),
            topic_tags=("研究生", "课程"),
        ),
        NoticeDraft(
            source_id="shu_main",
            canonical_url="https://www.shu.edu.cn/info/1051/400245.htm",
            title="关于调整校园管理的通告",
            normalized_text=(
                "学校通告，2026年8月30日至9月13日新生报到、开学典礼和军训期间，"
                "谢绝校外无关人员及车辆入校；因公入校的人员及车辆应经访客系统审批。"
                "军训日期为8月31日至9月13日，停在校内主干道的车辆须在8月29日18时前驶离。"
            ),
            published_at=datetime(2026, 8, 30, tzinfo=UTC),
            deadline_at=datetime(2026, 8, 29, 10, 0, tzinfo=UTC),
            deadline_confidence=DeadlineConfidence.HIGH,
            categories=("校园事务",),
        ),
        NoticeDraft(
            source_id="shu_main",
            canonical_url="https://www.shu.edu.cn/info/1051/395885.htm",
            title="关于宝山校区校内通行温馨提醒",
            normalized_text=(
                "2026年5月30日学校举办第五届校友返校乐跑活动。当天8时10分至8时45分，"
                "宝山校区外环线主干道机动车临时规避；8时45分前沿线不得停放车辆，"
                "校车首班只停靠南门和北门。"
            ),
            published_at=datetime(2026, 5, 29, tzinfo=UTC),
            categories=("校园事务",),
        ),
    )
    proof = Verification(ProofStatus.VERIFIED_SNAPSHOT, NOW, "snapshot:test-entry")
    for draft in drafts:
        repository.upsert(draft, verification=proof)
    repository.upsert(NoticeDraft(
        source_id="computer", canonical_url="https://demo.invalid/computer/demo-1",
        title="[演示]免修通知", normalized_text="演示免修内容",
        topic_tags=("synthetic_demo",),
    ))
    return NoticeService(
        repository,
        authority_by_source_id={"computer": 0.95, "shu_main": 1.0, "graduate": 0.9},
        now=lambda: NOW,
    )


def test_default_recommendations_retain_cards_without_keyword_hits(
    service: NoticeService,
) -> None:
    recommendations = service.recommend(NO_MATCH_QUERY)
    assert len(recommendations) == 3
    assert all(item.match_tier == "other" for item in recommendations)
    assert all(not item.reasons for item in recommendations)
    assert len(service.all()) == 3


def test_positive_minimum_relevance_is_an_opt_in_strict_filter(service: NoticeService) -> None:
    assert service.recommend(NoticeQuery(terms=NO_MATCH_QUERY.terms, minimum_relevance=0.1)) == []
    recommendations = service.recommend(NoticeQuery(terms=("免修",), minimum_relevance=0.5))
    assert len(recommendations) == 1
    assert recommendations[0].match_tier == "direct"
    assert service.recommend(NoticeQuery(terms=("免修",), minimum_relevance=1.0)) == []


def test_recommend_role_without_terms_returns_ranked_cards(service: NoticeService) -> None:
    persona = Persona(id="grad", label="研究生", education_level="研究生")
    recommendations = service.recommend(NoticeQuery(persona=persona))

    assert [item.match_tier for item in recommendations] == ["possible", "other", "other"]
    assert recommendations[0].notice.source_id == "graduate"
    assert recommendations[0].reasons == ("适用于研究生",)
    assert all(not item.reasons for item in recommendations[1:])
    assert len(service.recommend(NoticeQuery(persona=persona, limit=2))) == 2


def test_listing_title_hit_is_only_possible(service: NoticeService) -> None:
    saved = service.repository.upsert_listing(
        DiscoveredNotice(
            "graduate", "https://gs.shu.edu.cn/info/1029/200001.htm", "研究生科研项目报名通知",
        ),
        list_url="https://gs.shu.edu.cn/xwlb/sy.htm",
        checked_at=NOW,
    )

    recommendations = service.recommend(NoticeQuery(terms=("科研",)))

    assert len(recommendations) == 4
    assert recommendations[0].notice.id == saved.notice_id
    assert recommendations[0].match_tier == "possible"
    assert recommendations[0].reasons == ("匹配关键词：科研",)
    assert recommendations[0].notice.summary == ""
    assert recommendations[0].strong_reminder_eligible is False
    assert service.detail(saved.notice_id) is None


@pytest.mark.parametrize(
    ("terms", "old_level", "expected_tier", "expected_reasons"),
    [
        ((), "硕士生", "other", ()),
        (("科研",), "硕士生", "other", ()),
        (("课程",), "硕士生", "possible", ("匹配关键词：课程",)),
        (("课程",), "本科生", "possible", ("匹配关键词：课程",)),
    ],
)
def test_legacy_promoted_to_listing_uses_only_title_evidence(
    service: NoticeService, terms, old_level, expected_tier, expected_reasons,
) -> None:
    url = "https://gs.shu.edu.cn/info/1029/200002.htm"
    original = service.repository.upsert(NoticeDraft(
        source_id="graduate", canonical_url=url, title="旧正文通知",
        normalized_text="科研申请安排", education_levels=(old_level,),
        deadline_at=datetime(2026, 9, 23, tzinfo=UTC),
        deadline_confidence=DeadlineConfidence.HIGH,
    ))
    saved = service.repository.upsert_listing(
        DiscoveredNotice("graduate", url, "课程公开通知"),
        list_url="https://gs.shu.edu.cn/xwlb/sy.htm", checked_at=NOW,
    )
    persona = Persona(id="masters", label="硕士生", education_level="硕士生")

    recommendation = next(
        item for item in service.recommend(NoticeQuery(terms=terms, persona=persona))
        if item.notice.id == original.notice_id
    )

    assert saved.notice_id == original.notice_id
    assert recommendation.match_tier == expected_tier
    assert recommendation.reasons == expected_reasons
    assert recommendation.score.audience == 0.5
    assert recommendation.score.urgency == 0.0
    assert recommendation.strong_reminder_eligible is False
    assert service.detail(saved.notice_id) is None


def test_verified_service_fixture_matches_checked_official_articles(
    service: NoticeService,
) -> None:
    assert {(row.canonical_url, row.title) for row in service.all()} == {
        (
            "https://gs.shu.edu.cn/info/1029/173242.htm",
            "上海大学2026级硕博连读研究生、直博生 申请课程免修及成绩认定通知",
        ),
        ("https://www.shu.edu.cn/info/1051/400245.htm", "关于调整校园管理的通告"),
        ("https://www.shu.edu.cn/info/1051/395885.htm", "关于宝山校区校内通行温馨提醒"),
    }
    by_url = {row.canonical_url: row for row in service.all()}
    assert by_url["https://gs.shu.edu.cn/info/1029/173242.htm"].deadline_at.date() == date(
        2026, 10, 11
    )
    assert by_url["https://www.shu.edu.cn/info/1051/400245.htm"].published_at.date() == date(
        2026, 8, 30
    )
    assert by_url["https://www.shu.edu.cn/info/1051/395885.htm"].published_at.date() == date(
        2026, 5, 29
    )


def test_recommend_returns_explained_match_without_hiding_other_browse_modes(
    service: NoticeService,
) -> None:
    recommendations = service.recommend(NoticeQuery(terms=("免修",)))

    assert len(recommendations) == 3
    assert recommendations[0].notice.title == (
        "上海大学2026级硕博连读研究生、直博生 申请课程免修及成绩认定通知"
    )
    assert recommendations[0].match_tier == "direct"
    assert recommendations[0].reasons[0] == "匹配关键词：免修"
    assert len(service.latest()) == 3


def test_upcoming_and_detail_queries_are_deterministic(service: NoticeService) -> None:
    upcoming = service.upcoming()
    target = upcoming.expired[0]

    assert [notice.title for notice in upcoming.dated] == [
        "上海大学2026级硕博连读研究生、直博生 申请课程免修及成绩认定通知"
    ]
    assert [notice.title for notice in upcoming.expired] == ["关于调整校园管理的通告"]
    assert [notice.title for notice in upcoming.unknown] == ["关于宝山校区校内通行温馨提醒"]
    assert service.detail(target.id) == target
    assert service.detail("missing") is None


def test_every_service_read_hides_unverified_demo_notice(service: NoticeService) -> None:
    demo = next(row for row in service.repository.list_all() if "synthetic_demo" in row.topic_tags)
    assert all(row.id != demo.id for row in service.all())
    assert all(row.id != demo.id for row in service.latest())
    assert all(row.id != demo.id for row in service.upcoming().all_notices)
    recommendations = service.recommend(NoticeQuery(terms=("免修",)))
    assert all(item.notice.id != demo.id for item in recommendations)
    assert service.detail(demo.id) is None
    assert service.detail_evidence(demo.id) == {
        "duplicates": [], "possible_duplicates": [], "versions": []
    }


def test_service_read_without_policy_fails_closed(tmp_path: Path) -> None:
    database = Database(tmp_path / "unconfigured.sqlite3")
    database.initialize()
    service = NoticeService(NoticeRepository(database), authority_by_source_id={})
    with pytest.raises(RuntimeError, match="official notice policy"):
        service.all()


@pytest.fixture
def historical_notice(service):
    draft = NoticeDraft(
        source_id="graduate", canonical_url="https://gs.shu.edu.cn/info/1029/200003.htm",
        title="学位办理公开通知", normalized_text=HISTORICAL_PERSON_TABLE,
        published_at=NOW, deadline_at=datetime(2026, 9, 23, tzinfo=UTC),
        deadline_confidence=DeadlineConfidence.HIGH,
    )
    proof = Verification(ProofStatus.VERIFIED_SNAPSHOT, NOW, "snapshot:historical-synthetic")
    service.repository.upsert(draft, verification=proof)
    return service.repository.find_by_url(draft.canonical_url)


@pytest.mark.parametrize("mode", ["all", "latest", "upcoming", "recommend", "detail"])
def test_historical_person_table_is_projected_before_every_service_read(
    service, historical_notice, mode,
):
    if mode == "recommend":
        rows = [item.notice for item in service.recommend(NoticeQuery())]
    elif mode == "upcoming":
        rows = service.upcoming().all_notices
    elif mode == "detail":
        rows = [service.detail(historical_notice.id)]
    else:
        rows = getattr(service, mode)()

    actual = next(row for row in rows if row.id == historical_notice.id)
    assert actual == replace(historical_notice, summary="")
    assert actual is not historical_notice
    assert len(rows) == (1 if mode == "detail" else 4)
    assert service.repository.find_by_url(historical_notice.canonical_url) == historical_notice


def test_historical_person_table_cannot_supply_keyword_match_or_reason(service, historical_notice):
    query = NoticeQuery(terms=("样例甲",), minimum_relevance=0.1)
    assert service.recommend(query) == []
    recommendations = service.recommend(NoticeQuery(terms=("样例甲",)))
    assert len(recommendations) == 4
    assert all("样例甲" not in reason for item in recommendations for reason in item.reasons)
    assert next(item for item in recommendations
                if item.notice.id == historical_notice.id).match_tier == "other"


def test_historical_version_and_duplicate_previews_are_safe_without_database_writes(
    service, historical_notice,
):
    proof = Verification(ProofStatus.VERIFIED_SNAPSHOT, NOW, "snapshot:historical-synthetic")
    current_summary = HISTORICAL_PERSON_TABLE + " 已更新"
    draft = NoticeDraft(
        source_id=historical_notice.source_id, canonical_url=historical_notice.canonical_url,
        title=historical_notice.title, normalized_text=current_summary,
    )
    service.repository.upsert(draft, verification=proof)
    service.repository.upsert(replace(
        draft, canonical_url="https://gs.shu.edu.cn/info/1029/200004.htm",
    ), verification=proof)
    raw = service.repository.detail_evidence_verified(historical_notice.id)
    with service.repository.database.connect() as connection:
        before = tuple(connection.iterdump())

    evidence = service.detail_evidence(historical_notice.id)

    assert [version["summary"] for version in evidence["versions"]] == ["", ""]
    summary_change, = [change for change in evidence["versions"][-1]["changes"]
                       if change["field"] == "summary"]
    assert summary_change == {"field": "summary", "before": "", "after": ""}
    linked = evidence["duplicates"] + evidence["possible_duplicates"]
    assert linked
    assert all(item["notice"].summary == "" for item in linked)
    assert [version["fingerprint"] for version in evidence["versions"]] == [
        version["fingerprint"] for version in raw["versions"]
    ]
    assert [version["verified_at"] for version in evidence["versions"]] == [
        version["verified_at"] for version in raw["versions"]
    ]
    assert service.repository.detail_evidence_verified(historical_notice.id) == raw
    with service.repository.database.connect() as connection:
        assert tuple(connection.iterdump()) == before


@pytest.mark.parametrize("summary", [
    "表头依次为 序号 姓名 原学士学位证书编号，请于2026年10月提交材料，编号共16位。",
    "按序号填写姓名和原学士学位证书编号，第1项为申请流程，第2项为提交材料。",
    "学校获得科研项目立项，学生按通知提交材料。",
    "奖学金申请材料须于规定时间提交，录取手续按公开流程办理。",
])
def test_historical_ordinary_procedures_are_preserved(service, summary):
    saved = service.repository.upsert(NoticeDraft(
        source_id="graduate", canonical_url="https://gs.shu.edu.cn/info/1029/200005.htm",
        title="材料提交公开通知", normalized_text=summary,
    ), verification=Verification(ProofStatus.VERIFIED_SNAPSHOT, NOW, "snapshot:procedure"))
    assert service.detail(saved.notice_id).summary == summary


@pytest.mark.parametrize('tail', [
    '联系人：样例老师，电话：66124567-8001',
    '联系方式 示例甲 66124568*801',
    '组长：示例甲、示例乙',
    '办公电话：66124569*828',
])
def test_preserved_contact_tails_never_enter_cards_search_or_versions(service, tail):
    prefix = '2026年10月29日前提交推荐名单，GPA为3.5，按流程上传材料。'
    url = 'https://gs.shu.edu.cn/info/1029/200006.htm'
    proof = Verification(ProofStatus.VERIFIED_SNAPSHOT, NOW, 'snapshot:contact-synthetic')
    result = service.repository.upsert(
        NoticeDraft('graduate', url, '材料提交公开通知', prefix + tail),
        verification=proof, summary_override=prefix + tail,
    )
    service.repository.upsert(
        NoticeDraft('graduate', url, '材料提交公开通知', prefix + tail + ' 后续信息'),
        verification=proof, summary_override=prefix + tail + ' 后续信息',
    )
    with service.repository.database.connect() as connection:
        before = tuple(connection.iterdump())
    assert service.detail(result.notice_id).summary == prefix
    assert next(row for row in service.all() if row.id == result.notice_id).summary == prefix
    assert service.recommend(NoticeQuery(terms=('样例老师', '66124567'),
                                         minimum_relevance=.1)) == []
    evidence = service.detail_evidence(result.notice_id)
    assert all(version['summary'] == prefix for version in evidence['versions'])
    for version in evidence['versions']:
        for change in version.get('changes', []):
            if change['field'] == 'summary':
                assert change['before'] == prefix and change['after'] == prefix
    assert '样例老师' not in str(evidence) and '66124567' not in str(evidence)
    with service.repository.database.connect() as connection:
        assert tuple(connection.iterdump()) == before


@pytest.mark.parametrize('deadline', ['2026年10月29日', '2026-10-29', '20261029'])
def test_unlabelled_landline_is_masked_without_losing_procedure_or_grade(service, deadline):
    summary = deadline + '前提交，GPA为3.5，材料咨询请拨打66124567-8001。'
    result = service.repository.upsert(
        NoticeDraft('graduate', 'https://gs.shu.edu.cn/info/1029/200007.htm',
                    '材料提交公开通知', summary),
        verification=Verification(ProofStatus.VERIFIED_SNAPSHOT, NOW, 'snapshot:landline'),
    )
    projected = service.detail(result.notice_id).summary
    assert '66124567' not in projected
    assert deadline in projected and 'GPA为3.5' in projected


def test_historical_title_and_contact_identifiers_are_redacted(service):
    saved = service.repository.upsert(NoticeDraft(
        source_id="graduate", canonical_url="https://gs.shu.edu.cn/info/1029/200006.htm",
        title="关于学生样例甲同学获得奖励的通知",
        normalized_text="申请流程请联系13812345678或sample@example.invalid。",
    ), verification=Verification(ProofStatus.VERIFIED_SNAPSHOT, NOW, "snapshot:contacts"))

    detail = service.detail(saved.notice_id)

    assert "样例甲" not in detail.title
    assert detail.summary == "申请流程请联系[电话已隐藏]或[邮箱已隐藏]。"


@pytest.mark.parametrize(
    ("persona", "expected_total"),
    [
        (None, 0.9),
        (Persona(id="grad", label="研究生", education_level="研究生", grade="不限"), 1.0),
    ],
)
def test_recommend_uses_loaded_weights_for_profile_mode(
    service: NoticeService,
    tmp_path: Path,
    persona: Persona | None,
    expected_total: float,
) -> None:
    config_path = tmp_path / "ranking.toml"
    config_path.write_text(
        "[profile]\n"
        "relevance = 0.0\n"
        "audience = 1.0\n"
        "urgency = 0.0\n"
        "authority = 0.0\n"
        "freshness = 0.0\n"
        "[no_profile]\n"
        "relevance = 0.0\n"
        "urgency = 0.0\n"
        "authority = 1.0\n"
        "freshness = 0.0\n",
        encoding="utf-8",
    )
    configured = NoticeService(
        service.repository,
        authority_by_source_id={"graduate": 0.9},
        ranking_weights=load_ranking_weights(config_path),
        now=lambda: NOW,
    )

    recommendations = configured.recommend(NoticeQuery(terms=("免修",), persona=persona))

    assert len(recommendations) == 3
    assert recommendations[0].score.total == pytest.approx(expected_total)


def test_default_expansion_does_not_promote_administrative_summary_over_real_opportunity(service):
    from shu_insight.processing.classify import load_taxonomy
    from shu_insight.ranking.expand import Query, expand_terms

    proof = Verification(ProofStatus.VERIFIED_SNAPSHOT, NOW, "snapshot:synthetic-ranking")
    saved_ids = []
    for number, title, body in (
        (300001, "行政情况通报", "科研、学术、论文、实验室日常管理情况。"),
        (300002, "事务情况说明", "学术和论文办理流程说明。"),
        (300003, "推免申请通知", "本科生按公开安排提交推免申请材料。"),
    ):
        saved_ids.append(service.repository.upsert(NoticeDraft(
            source_id="computer", canonical_url=f"https://cs.shu.edu.cn/info/1/{number}.htm",
            title=title, normalized_text=body, published_at=NOW,
        ), verification=proof).notice_id)
    expanded = expand_terms(
        Query(tags=("推免", "保研", "竞赛", "科研")),
        load_taxonomy(Path("config/taxonomy.toml")),
    )
    ranked = service.recommend(NoticeQuery(
        terms=expanded.all_terms,
        persona=Persona(id="p", label="本科大二计科", education_level="本科生", grade="大二",
                        college="计算机工程与科学学院", major="计算机科学与技术"),
    ))
    assert len(ranked) == 6
    assert ranked[0].notice.id == saved_ids[2]
    by_id = {item.notice.id: item for item in ranked}
    assert [by_id[item_id].match_tier for item_id in saved_ids] == [
        "possible", "possible", "direct",
    ]
    assert "学术" in by_id[saved_ids[0]].reasons[0]
    assert "论文" in by_id[saved_ids[1]].reasons[0]
