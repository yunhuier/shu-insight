from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from shu_insight.domain.models import (
    DeadlineConfidence,
    NoticeRecord,
    ParseState,
    Persona,
    ProofStatus,
)
from shu_insight.ranking.explain import explain
from shu_insight.ranking.relevance import RelevanceResult
from shu_insight.ranking.score import (
    ScoreInputs,
    can_strong_remind,
    deadline_timeline,
    load_ranking_weights,
    rank_notices,
    score_no_profile,
    score_profile,
)

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


def test_ranking_config_loads_both_weight_sets() -> None:
    weights = load_ranking_weights(Path("config/ranking.toml"))

    assert weights.profile.audience == 0.25
    assert weights.no_profile.audience == 0.0


def notice(
    notice_id: str,
    *,
    published_days_ago: int = 1,
    deadline_days_ahead: int | None = 5,
    majors: tuple[str, ...] = (),
    levels: tuple[str, ...] = (),
) -> NoticeRecord:
    deadline = NOW + timedelta(days=deadline_days_ahead) if deadline_days_ahead else None
    return NoticeRecord(
        id=notice_id,
        source_id="computer",
        canonical_url=f"https://cs.shu.edu.cn/info/1/{notice_id}.htm",
        title=f"通知 {notice_id}",
        summary="推免报名安排",
        published_at=NOW - timedelta(days=published_days_ago),
        deadline_at=deadline,
        deadline_confidence=DeadlineConfidence.HIGH,
        education_levels=levels,
        majors=majors,
        categories=("升学推免",),
        topic_tags=("升学",),
    )


def test_profile_score_uses_spec_weights() -> None:
    parts = ScoreInputs(
        relevance=0.8,
        audience=0.6,
        urgency=0.5,
        authority=1.0,
        freshness=0.4,
    )

    assert score_profile(parts) == pytest.approx(
        0.40 * 0.8 + 0.25 * 0.6 + 0.20 * 0.5 + 0.10 * 1 + 0.05 * 0.4
    )


def test_no_profile_score_uses_no_identity_weighting() -> None:
    parts = ScoreInputs(
        relevance=0.8,
        audience=0.0,
        urgency=0.5,
        authority=1.0,
        freshness=0.4,
    )

    assert score_no_profile(parts) == pytest.approx(
        0.55 * 0.8 + 0.25 * 0.5 + 0.10 * 1 + 0.10 * 0.4
    )


def test_rank_notices_returns_all_and_breaks_ties_by_newest() -> None:
    older = notice("older", published_days_ago=4)
    newer = notice("newer", published_days_ago=1)
    relevance = {
        item.id: RelevanceResult(0.8, frozenset({"推免"}), "exact")
        for item in (older, newer)
    }

    ranked = rank_notices(
        [older, newer],
        relevance,
        {"computer": 0.95},
        NOW,
        persona=None,
    )

    assert [item.notice.id for item in ranked] == ["newer", "older"]
    assert len(ranked) == 2


def test_major_mismatch_reduces_score_but_never_filters_notice() -> None:
    item = notice("major", majors=("人工智能",))
    persona = Persona(
        id="p",
        label="计科同学",
        education_level="本科生",
        grade="大二",
        college="计算机工程与科学学院",
        major="计算机科学与技术",
    )

    ranked = rank_notices(
        [item],
        {item.id: RelevanceResult(0.7, frozenset({"推免"}), "exact")},
        {"computer": 0.95},
        NOW,
        persona=persona,
    )

    assert len(ranked) == 1
    assert ranked[0].score.audience == 0.0
    assert "适用于计算机科学与技术" not in explain(ranked[0])


def test_explanation_mentions_only_evidenced_nonzero_components() -> None:
    item = notice("explain", levels=("本科生",))
    persona = Persona(
        id="p",
        label="本科生",
        education_level="本科生",
        grade="不限",
    )
    recommendation = rank_notices(
        [item],
        {item.id: RelevanceResult(0.8, frozenset({"推免"}), "exact")},
        {"computer": 0.95},
        NOW,
        persona=persona,
    )[0]

    explanation = explain(recommendation)
    assert "匹配关键词：推免" in explanation
    assert "适用于本科生" in explanation
    assert "距离截止时间 5 天" in explanation


def test_low_date_only_deadline_last_through_shanghai_calendar_day() -> None:
    shanghai = ZoneInfo("Asia/Shanghai")
    item = replace(notice("date-only"),
                   deadline_at=datetime(2026, 10, 11, tzinfo=shanghai),
                   deadline_confidence=DeadlineConfidence.LOW)
    for instant in (datetime(2026, 10, 11, 0, 1, tzinfo=shanghai),
                    datetime(2026, 10, 11, 23, 59, tzinfo=shanghai)):
        assert deadline_timeline([item], instant).dated == [item]
        assert can_strong_remind(item, instant) is False
    assert deadline_timeline(
        [item], datetime(2026, 10, 12, tzinfo=shanghai)
    ).expired == [item]


def test_unknown_audience_is_not_explicit_match() -> None:
    from shu_insight.ranking.score import audience_evidence

    item = replace(notice("unknown"), deadline_at=None)
    persona = Persona(id="p", label="本科生", education_level="本科生")

    assert audience_evidence(item, persona) == ((), False)
    recommendation = rank_notices([item], {}, {}, NOW, persona=persona)[0]
    assert recommendation.match_tier == "other"
    assert recommendation.reasons == ()
    assert recommendation.score.audience == 0.5


@pytest.mark.parametrize(
    ("field", "selected"),
    [("education_levels", "研究生"), ("grades", "大三"),
     ("colleges", "理学院"), ("majors", "人工智能")],
)
def test_explicit_audience_conflict_forces_other_despite_title_hit(field, selected) -> None:
    from shu_insight.ranking.score import audience_evidence

    item = replace(notice("conflict"), title="推免报名", deadline_at=None, **{field: (selected,)})
    persona = Persona(id="p", label="本科生", education_level="本科生", grade="大二",
                      college="计算机工程与科学学院", major="计算机科学与技术")

    assert audience_evidence(item, persona) == ((), True)
    recommendation = rank_notices(
        [item], {item.id: RelevanceResult(1.0, frozenset({"推免"}), "exact")},
        {}, NOW, persona=persona,
    )[0]
    assert recommendation.match_tier == "other"
    assert recommendation.reasons == ("匹配关键词：推免",)


def test_audience_match_does_not_cancel_another_explicit_conflict() -> None:
    from shu_insight.ranking.score import audience_evidence

    item = replace(notice("mixed", levels=("本科生",), majors=("人工智能",)), deadline_at=None)
    persona = Persona(id="p", label="计科", education_level="本科生", major="计算机科学与技术")
    assert audience_evidence(item, persona) == (("适用于本科生",), True)
    recommendation = rank_notices([item], {}, {}, NOW, persona=persona)[0]
    assert recommendation.match_tier == "other"


@pytest.mark.parametrize("term", ["实验室", "科研", "学术", "论文"])
def test_generic_summary_hit_is_only_possible(term: str) -> None:
    item = replace(notice("generic"), summary=f"{term}开放安排", deadline_at=None)
    recommendation = rank_notices(
        [item], {item.id: RelevanceResult(1.0, frozenset({term}), "exact")},
        {}, NOW, persona=None,
    )[0]
    assert recommendation.match_tier == "possible"
    assert recommendation.reasons == (f"匹配关键词：{term}",)


@pytest.mark.parametrize(
    ("title", "summary", "term"),
    [("科研项目通知", "", "科研"), ("申请通知", "推免申请安排", "推免")],
)
def test_parsed_title_or_specific_summary_hit_is_direct(title, summary, term) -> None:
    item = replace(notice("direct"), title=title, summary=summary, deadline_at=None)
    recommendation = rank_notices(
        [item], {item.id: RelevanceResult(1.0, frozenset({term}), "exact")},
        {}, NOW, persona=None,
    )[0]
    assert recommendation.match_tier == "direct"
    assert recommendation.reasons == (f"匹配关键词：{term}",)


def test_list_only_summary_is_not_body_evidence() -> None:
    item = replace(notice("listing"), title="通知", summary="推免申请", deadline_at=None,
                   parse_state=ParseState.LISTING_ONLY)
    recommendation = rank_notices(
        [item], {item.id: RelevanceResult(1.0, frozenset({"推免"}), "exact")},
        {}, NOW, persona=None,
    )[0]
    assert recommendation.match_tier == "other"
    assert recommendation.reasons == ()


@pytest.mark.parametrize("old_level", ["硕士生", "本科生"])
@pytest.mark.parametrize("title_hit", [False, True])
def test_list_only_stale_audience_is_unknown_even_with_title_hit(old_level, title_hit) -> None:
    from shu_insight.ranking.score import audience_evidence

    item = replace(
        notice("stale-audience", levels=(old_level,)), title="课程公开通知",
        deadline_at=None, parse_state=ParseState.LISTING_ONLY,
    )
    persona = Persona(id="masters", label="硕士生", education_level="硕士生")
    relevance = {item.id: RelevanceResult(1.0, frozenset({"课程"}), "exact")} if title_hit else {}

    recommendation, = rank_notices([item], relevance, {}, NOW, persona=persona)

    assert audience_evidence(item, persona) == ((), False)
    assert recommendation.score.audience == 0.5
    assert recommendation.match_tier == ("possible" if title_hit else "other")
    assert recommendation.reasons == (("匹配关键词：课程",) if title_hit else ())


def test_list_only_stale_deadline_cannot_supply_reminder_or_urgency() -> None:
    item = replace(notice("stale-deadline"), parse_state=ParseState.LISTING_ONLY)

    recommendation, = rank_notices([item], {}, {}, NOW, persona=None)

    assert can_strong_remind(item, NOW) is False
    assert recommendation.strong_reminder_eligible is False
    assert recommendation.score.urgency == 0.0
    assert recommendation.reasons == ()


def test_category_or_tag_hit_alone_is_not_personal_evidence() -> None:
    item = replace(notice("tags"), summary="提交材料", deadline_at=None)
    recommendation = rank_notices(
        [item], {item.id: RelevanceResult(1.0, frozenset({"升学"}), "exact")},
        {}, NOW, persona=None,
    )[0]
    assert recommendation.match_tier == "other"
    assert recommendation.reasons == ()


def test_direct_precedes_other_even_when_other_is_newer() -> None:
    direct = replace(notice("direct", published_days_ago=100), title="推免报名", deadline_at=None)
    possible = replace(notice("possible"), summary="科研开放安排", deadline_at=None)
    other = notice("other", published_days_ago=0, deadline_days_ahead=1)
    ranked = rank_notices(
        [other, possible, direct],
        {"direct": RelevanceResult(0.1, frozenset({"推免"}), "exact"),
         "possible": RelevanceResult(0.1, frozenset({"科研"}), "exact")},
        {"computer": 1.0}, NOW, persona=None,
    )
    assert [item.notice.id for item in ranked] == ["direct", "possible", "other"]
    assert [item.match_tier for item in ranked] == ["direct", "possible", "other"]
    assert ranked[2].score.total > ranked[0].score.total


def test_expired_result_has_no_strong_reminder() -> None:
    item = replace(notice("expired", deadline_days_ahead=-1), title="推免报名")
    recommendation = rank_notices(
        [item], {item.id: RelevanceResult(1.0, frozenset({"推免"}), "exact")},
        {}, NOW, persona=None,
    )[0]
    assert recommendation.match_tier == "direct"
    assert recommendation.strong_reminder_eligible is False
    assert recommendation.score.urgency == 0.0
    assert recommendation.reasons == ("匹配关键词：推免",)


@pytest.mark.parametrize("state", [ParseState.PARSED, ParseState.LISTING_ONLY])
@pytest.mark.parametrize(
    ("title", "summary"),
    [
        ("本科新生照片采集通知", ""),
        ("本科毕业班材料提交事项", ""),
        ("本科应届毕业生信息核对", ""),
        ("关于毕业班及延长学籍学生游泳公选课的选课通知", ""),
        ("教师教学竞赛报名通知", ""),
        ("关于推荐教师参与第二届区域高校示例杯智慧教学大赛的通知", ""),
        ("竞赛报名通知", "参赛对象：仅限本校教师。"),
    ],
)
def test_specific_audience_restriction_overrides_legacy_broad_student_metadata(
    state, title, summary,
) -> None:
    item = replace(notice("restricted", levels=("本科生",)), title=title, summary=summary,
                   parse_state=state, deadline_at=None, proof_status=ProofStatus.VERIFIED_SNAPSHOT)
    persona = Persona(id="p", label="本科大二", education_level="本科生", grade="大二")
    recommendation, = rank_notices(
        [item], {item.id: RelevanceResult(1.0, frozenset({"通知", "竞赛", "事项", "信息"}),
                                         "exact")}, {}, NOW, persona=persona,
    )

    # A list title is evidence, but the old body and metadata are not.
    if state is ParseState.LISTING_ONLY and summary:
        assert recommendation.match_tier == "possible"
        assert recommendation.score.audience == 0.5
    else:
        assert recommendation.match_tier == "other"
        assert recommendation.score.audience == 0.0


@pytest.mark.parametrize(
    ("title", "summary"),
    [
        ("师生竞赛报名通知", "面向全校教师和学生开放报名。"),
        ("学生科研讲座通知", "面向本科生，主讲人为学院教师。"),
        ("教师与学生共同参加的竞赛通知", "欢迎师生参与。"),
        ("学生竞赛通知", "指导教师参加教学竞赛的经验将用于辅导学生。"),
        ("学生讲座通知", "教师介绍本科新生照片采集的工作经验。"),
        ("教师指导学生参加比赛通知", "面向本科生开放。"),
        ("关于推荐教师和学生参与区域高校智慧教学大赛的通知", "欢迎师生参与。"),
    ],
)
def test_incidental_teacher_mentions_and_mixed_events_do_not_exclude_students(title, summary):
    item = replace(notice("inclusive"), title=title, summary=summary, deadline_at=None,
                   proof_status=ProofStatus.VERIFIED_SNAPSHOT)
    persona = Persona(id="p", label="本科大二", education_level="本科生", grade="大二")
    recommendation, = rank_notices(
        [item], {item.id: RelevanceResult(1.0, frozenset({"通知"}), "exact")},
        {}, NOW, persona=persona,
    )
    assert recommendation.match_tier == "direct"


@pytest.mark.parametrize(
    ("summary", "grade"),
    [
        pytest.param("面向教师征集指导项目，面向学生开放竞赛报名。", "大二",
                     id="separate-positive-student-clause"),
        pytest.param("参赛对象：本科新生和教师。", "大一", id="new-students-and-teachers"),
        pytest.param("参赛对象：学生和教师。", "大二", id="students-and-teachers"),
    ],
)
def test_explicit_student_participation_keeps_mixed_competition_direct(summary, grade):
    item = replace(notice("mixed-participation"), title="竞赛报名通知", summary=summary,
                   deadline_at=None, proof_status=ProofStatus.VERIFIED_SNAPSHOT)
    persona = Persona(id="p", label="本科生", education_level="本科生", grade=grade)
    recommendation, = rank_notices(
        [item], {item.id: RelevanceResult(1.0, frozenset({"竞赛"}), "exact")},
        {}, NOW, persona=persona,
    )
    assert recommendation.match_tier == "direct"
    assert recommendation.score.audience > 0.0


@pytest.mark.parametrize("summary", [
    "仅面向教师，不面向学生。",
    "面向教师征集项目，禁止面向学生开放报名。",
    "面向教师征集指导项目，主讲教师介绍学生比赛经验。",
])
def test_negative_or_incidental_student_mentions_do_not_cancel_teacher_scope(summary):
    item = replace(notice("teacher-only"), title="竞赛报名通知", summary=summary,
                   deadline_at=None, proof_status=ProofStatus.VERIFIED_SNAPSHOT)
    recommendation, = rank_notices(
        [item], {item.id: RelevanceResult(1.0, frozenset({"竞赛"}), "exact")}, {}, NOW,
        persona=Persona(id="p", label="本科大二", education_level="本科生", grade="大二"),
    )
    assert recommendation.match_tier == "other"
    assert recommendation.score.audience == 0.0


@pytest.mark.parametrize(
    ("state", "proof"),
    [(ParseState.LISTING_ONLY, ProofStatus.VERIFIED_LISTING),
     (ParseState.PARSED, ProofStatus.UNVERIFIED_LEGACY)],
)
def test_unverified_student_summary_cannot_cancel_teacher_title_scope(state, proof):
    item = replace(notice("teacher-title"), title="教师教学竞赛报名通知",
                   summary="面向学生开放竞赛报名。", deadline_at=None,
                   parse_state=state, proof_status=proof)
    recommendation, = rank_notices(
        [item], {item.id: RelevanceResult(1.0, frozenset({"竞赛"}), "exact")}, {}, NOW,
        persona=Persona(id="p", label="本科大二", education_level="本科生", grade="大二"),
    )
    assert recommendation.match_tier == "other"
    assert recommendation.score.audience == 0.0


@pytest.mark.parametrize(
    ("title", "summary", "level", "grade", "expected_tier"),
    [
        ("本科新生照片采集通知", "", "本科生", "大一", "direct"),
        ("本科毕业班材料通知", "", "本科生", "大四", "direct"),
        ("本科新生照片采集通知", "", "不限", "不限", "direct"),
        ("教师教学竞赛通知", "", "不限", "不限", "direct"),
        ("关于毕业班及延长学籍学生选课通知", "", "不限", "不限", "direct"),
        ("关于毕业班及延长学籍学生选课通知", "", "本科生", "不限", "direct"),
        ("关于毕业班及延长学籍学生选课通知", "", "本科生", "大三", "direct"),
        ("关于推荐教师参与区域高校教学大赛的通知", "", "不限", "不限", "direct"),
        ("照片采集通知", "面向本科新生办理。", "本科生", "大二", "other"),
        ("材料提交通知", "适用对象：本科毕业班学生。", "本科生", "大二", "other"),
    ],
)
def test_scope_uses_explicit_summary_but_does_not_invent_conflict_for_unknown_persona(
    title, summary, level, grade, expected_tier,
):
    item = replace(notice("scope"), title=title, summary=summary, deadline_at=None,
                   proof_status=ProofStatus.VERIFIED_SNAPSHOT)
    recommendation, = rank_notices(
        [item], {item.id: RelevanceResult(1.0, frozenset({"通知"}), "exact")},
        {}, NOW, persona=Persona(id="p", label="样例角色", education_level=level, grade=grade),
    )
    assert recommendation.match_tier == expected_tier


def test_unverified_summary_cannot_create_an_audience_restriction():
    from shu_insight.ranking.score import audience_evidence

    item = replace(notice("unverified"), title="竞赛通知", summary="参赛对象：本校教师。",
                   proof_status=ProofStatus.UNVERIFIED_LEGACY)
    persona = Persona(id="p", label="本科大二", education_level="本科生", grade="大二")
    assert audience_evidence(item, persona) == ((), False)


@pytest.mark.parametrize("tier", ["direct", "possible", "other"])
@pytest.mark.parametrize("stale_kind", ["expired", "old"])
def test_current_records_precede_higher_scoring_stale_records_in_each_tier(tier, stale_kind):
    title = "推免报名" if tier == "direct" else "一般通知"
    summary = "科研安排" if tier == "possible" else "材料安排"
    current = replace(notice("current"), title=title, summary=summary, deadline_at=None)
    stale = replace(current, id="stale", published_at=NOW - timedelta(days=91)
                    if stale_kind == "old" else NOW,
                    deadline_at=NOW - timedelta(days=1) if stale_kind == "expired" else None)
    term = "推免" if tier == "direct" else "科研"
    relevance = {
        "current": RelevanceResult(0.01, frozenset({term}) if tier != "other"
                                   else frozenset(), "exact"),
        "stale": RelevanceResult(1.0, frozenset({term}) if tier != "other"
                                 else frozenset(), "exact"),
    }
    ranked = rank_notices([stale, current], relevance, {}, NOW, persona=None)
    assert [item.notice.id for item in ranked] == ["current", "stale"]
    assert [item.match_tier for item in ranked] == [tier, tier]
    assert ranked[0].score.total < ranked[1].score.total


@pytest.mark.parametrize("published_at", [None, NOW + timedelta(days=2)])
def test_unknown_or_future_publication_is_not_invented_as_old(published_at):
    item = replace(notice("unknown-date"), title="推免报名", published_at=published_at,
                   deadline_at=None)
    old = replace(item, id="old", published_at=NOW - timedelta(days=91))
    ranked = rank_notices(
        [old, item], {old.id: RelevanceResult(1.0, frozenset({"推免"}), "exact"),
                      item.id: RelevanceResult(0.01, frozenset({"推免"}), "exact")},
        {}, NOW, persona=None,
    )
    assert [row.notice.id for row in ranked] == ["unknown-date", "old"]
    assert ranked[0].score.freshness == 0.0


def test_exactly_ninety_days_and_listing_stale_deadlines_do_not_trigger_demotion():
    recent = replace(notice("recent"), title="科研通知", summary="", deadline_at=None)
    ninety = replace(recent, id="ninety", published_at=NOW - timedelta(days=90))
    listing = replace(recent, id="listing", parse_state=ParseState.LISTING_ONLY,
                      deadline_at=NOW - timedelta(days=1))
    second_listing = replace(listing, id="second-listing", deadline_at=None)
    ranked = rank_notices(
        [recent, ninety, listing, second_listing],
        {item_id: RelevanceResult(score, frozenset({"科研"}), "exact")
         for item_id, score in [("recent", 0.01), ("ninety", 1.0),
                                ("listing", 1.0), ("second-listing", 0.01)]},
        {}, NOW, persona=None,
    )
    assert [row.notice.id for row in ranked] == ["ninety", "recent", "listing", "second-listing"]
