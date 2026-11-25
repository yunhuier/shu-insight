from datetime import UTC, datetime

from shu_insight.domain.models import DeadlineConfidence, NoticeRecord
from shu_insight.mining.versioning import compare_versions


def version(*, deadline_day: int, location: str, fingerprint: str) -> NoticeRecord:
    return NoticeRecord(
        id="notice-1",
        source_id="computer",
        canonical_url="https://cs.shu.edu.cn/info/1/2.htm",
        title="创新赛报名通知",
        summary="请按要求完成报名。",
        published_at=datetime(2026, 9, 20, tzinfo=UTC),
        deadline_at=datetime(2026, 9, deadline_day, 18, 0, tzinfo=UTC),
        deadline_confidence=DeadlineConfidence.HIGH,
        location=location,
        categories=("竞赛创新",),
        topic_tags=("竞赛",),
        content_fingerprint=fingerprint,
    )


OLD_NOTICE = version(deadline_day=28, location="宝山校区A楼", fingerprint="old-fingerprint")
UPDATED_NOTICE = version(deadline_day=30, location="宝山校区B楼", fingerprint="new-fingerprint")


def test_version_diff_records_deadline_and_location_changes() -> None:
    diff = compare_versions(OLD_NOTICE, UPDATED_NOTICE)

    assert diff.changed_fields == {"deadline_at", "location"}
    assert diff.previous_fingerprint != diff.current_fingerprint
    assert {change.field for change in diff.changes} == {"deadline_at", "location"}
    assert next(change for change in diff.changes if change.field == "location").before == (
        "宝山校区A楼"
    )


def test_unchanged_version_has_no_field_changes() -> None:
    diff = compare_versions(OLD_NOTICE, OLD_NOTICE)

    assert diff.changed_fields == set()
    assert diff.has_changes is False
