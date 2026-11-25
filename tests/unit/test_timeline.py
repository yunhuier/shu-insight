from __future__ import annotations

from datetime import UTC, datetime, timedelta

from shu_insight.domain.models import DeadlineConfidence, NoticeRecord
from shu_insight.ranking.score import (
    can_strong_remind,
    deadline_timeline,
    latest_timeline,
)

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


def notice(
    notice_id: str,
    *,
    published_at: datetime | None = NOW,
    deadline_at: datetime | None = None,
    confidence: DeadlineConfidence = DeadlineConfidence.UNKNOWN,
) -> NoticeRecord:
    return NoticeRecord(
        id=notice_id,
        source_id="computer",
        canonical_url=f"https://cs.shu.edu.cn/info/1/{notice_id}.htm",
        title=notice_id,
        summary="通知摘要",
        published_at=published_at,
        deadline_at=deadline_at,
        deadline_confidence=confidence,
    )


FUTURE_NOTICE = notice(
    "future",
    deadline_at=NOW + timedelta(days=3),
    confidence=DeadlineConfidence.HIGH,
)
UNKNOWN_NOTICE = notice("unknown")
LOW_CONFIDENCE_NOTICE = notice(
    "low",
    deadline_at=NOW + timedelta(days=3),
    confidence=DeadlineConfidence.LOW,
)
PAST_DEADLINE_NOTICE = notice(
    "past",
    deadline_at=NOW - timedelta(days=1),
    confidence=DeadlineConfidence.HIGH,
)


def test_low_confidence_or_past_deadline_cannot_strong_remind() -> None:
    assert can_strong_remind(LOW_CONFIDENCE_NOTICE, NOW) is False
    assert can_strong_remind(PAST_DEADLINE_NOTICE, NOW) is False
    assert can_strong_remind(FUTURE_NOTICE, NOW) is True


def test_deadline_timeline_separates_unknown_and_expired_deadlines() -> None:
    timeline = deadline_timeline(
        [UNKNOWN_NOTICE, PAST_DEADLINE_NOTICE, FUTURE_NOTICE], NOW
    )

    assert timeline.dated == [FUTURE_NOTICE]
    assert timeline.unknown == [UNKNOWN_NOTICE]
    assert timeline.expired == [PAST_DEADLINE_NOTICE]
    assert timeline.all_notices == [FUTURE_NOTICE, PAST_DEADLINE_NOTICE, UNKNOWN_NOTICE]


def test_latest_timeline_sorts_dated_newest_first_and_keeps_unknown() -> None:
    older = notice("older", published_at=NOW - timedelta(days=5))
    newer = notice("newer", published_at=NOW - timedelta(days=1))
    unknown = notice("unknown-published", published_at=None)

    timeline = latest_timeline([older, unknown, newer])

    assert timeline.dated == [newer, older]
    assert timeline.unknown == [unknown]
    assert timeline.all_notices == [newer, older, unknown]
