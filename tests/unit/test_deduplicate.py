from __future__ import annotations

from datetime import UTC, datetime

from shu_insight.domain.models import DeadlineConfidence, NoticeRecord
from shu_insight.mining.deduplicate import assign_clusters, duplicate_similarity


def notice(
    notice_id: str,
    source_id: str,
    url: str,
    title: str,
    *,
    published_at: datetime = datetime(2026, 9, 20, tzinfo=UTC),
    summary_simhash: int = 0xAA55AA55AA55AA55,
    categories: tuple[str, ...] = ("竞赛创新",),
    topic_tags: tuple[str, ...] = ("竞赛", "人工智能"),
) -> NoticeRecord:
    return NoticeRecord(
        id=notice_id,
        source_id=source_id,
        canonical_url=url,
        title=title,
        summary="面向全校学生举办人工智能创新竞赛，优秀项目可进入科研训练。",
        published_at=published_at,
        deadline_at=datetime(2026, 9, 30, tzinfo=UTC),
        deadline_confidence=DeadlineConfidence.HIGH,
        location="宝山校区",
        categories=categories,
        topic_tags=topic_tags,
        summary_simhash=summary_simhash,
    )


BKSY_COMPETITION = notice(
    "bksy-1",
    "undergraduate",
    "https://bksy.shu.edu.cn/info/1/100.htm",
    "关于举办校园人工智能创新赛的通知",
)
GS_PARAPHRASED_COMPETITION = notice(
    "gs-1",
    "graduate",
    "https://gs.shu.edu.cn/info/2/200.htm",
    "校园人工智能创新竞赛报名通知",
)


def test_cross_source_repost_joins_cluster_but_keeps_both_urls() -> None:
    clusters = assign_clusters(
        [BKSY_COMPETITION, GS_PARAPHRASED_COMPETITION], threshold=0.78
    )

    assert len(clusters) == 1
    assert {notice.canonical_url for notice in clusters[0].members} == {
        BKSY_COMPETITION.canonical_url,
        GS_PARAPHRASED_COMPETITION.canonical_url,
    }
    assert clusters[0].evidence[0].score >= 0.78


def test_exact_canonical_url_is_always_a_duplicate() -> None:
    updated = notice(
        "bksy-2",
        "undergraduate",
        BKSY_COMPETITION.canonical_url,
        "标题经过轻微更新",
        summary_simhash=0,
        categories=("校园事务",),
        topic_tags=(),
    )

    evidence = duplicate_similarity(BKSY_COMPETITION, updated)

    assert evidence.score == 1.0
    assert evidence.exact_match is True


def test_uncertain_pair_is_flagged_but_not_merged() -> None:
    uncertain = notice(
        "main-2",
        "shu_main",
        "https://www.shu.edu.cn/info/3/300.htm",
        "人工智能创新项目申报说明",
        summary_simhash=0xAA55AA55AA55FF00,
        topic_tags=("科研", "人工智能"),
    )

    assignments = assign_clusters([BKSY_COMPETITION, uncertain], threshold=0.90)

    assert len(assignments) == 2
    assert len(assignments.possible_duplicates) == 1
    assert assignments.possible_duplicates[0].disposition == "possible_duplicate"


def test_same_keyword_notices_outside_ninety_days_are_not_candidates() -> None:
    old = notice(
        "old",
        "shu_main",
        "https://www.shu.edu.cn/info/4/400.htm",
        "人工智能竞赛往届回顾",
        published_at=datetime(2025, 1, 1, tzinfo=UTC),
    )

    assignments = assign_clusters([BKSY_COMPETITION, old], threshold=0.70)

    assert len(assignments) == 2
    assert assignments.possible_duplicates == ()
