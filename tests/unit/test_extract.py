from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from shu_insight.domain.models import DeadlineConfidence
from shu_insight.processing.extract import extract_fields
from shu_insight.processing.normalize import NormalizedDocument

SH_TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 9, 21, 12, 0, tzinfo=SH_TZ)


def document(text: str) -> NormalizedDocument:
    return NormalizedDocument(text=text, paragraphs=tuple(text.splitlines()))


def test_deadline_nearest_cue_wins_and_all_values_keep_evidence() -> None:
    text = (
        "会议时间为2026年9月22日。报名截止：2026年9月30日18:00。\n"
        "地点：宝山校区东区一号楼。面向本科生、大二学生以及计算机工程与科学学院人工智能专业。\n"
        "提交材料：项目说明书、演示视频。"
    )

    fields = extract_fields(document(text), NOW)

    assert fields.deadline_at == datetime(2026, 9, 30, 18, 0, tzinfo=SH_TZ)
    assert fields.deadline_confidence is DeadlineConfidence.HIGH
    assert fields.strong_reminder_eligible is True
    assert fields.selected_deadline is not None
    assert fields.selected_deadline.evidence_text == "2026年9月30日18:00"
    assert text[fields.selected_deadline.start : fields.selected_deadline.end] == (
        fields.selected_deadline.evidence_text
    )
    assert [item.value for item in fields.locations] == ["宝山校区东区一号楼"]
    assert {item.value for item in fields.education_levels} == {"本科生"}
    assert {item.value for item in fields.grades} == {"大二"}
    assert {item.value for item in fields.colleges} == {"计算机工程与科学学院"}
    assert {item.value for item in fields.majors} == {"人工智能"}
    assert {item.value for item in fields.materials} == {"项目说明书", "演示视频"}
    assert all(item.evidence_text for item in fields.all_evidence())


def test_month_day_deadline_infers_current_year_with_medium_confidence() -> None:
    fields = extract_fields(document("报名截止：10月2日 18:00"), NOW)

    assert fields.deadline_at == datetime(2026, 10, 2, 18, 0, tzinfo=SH_TZ)
    assert fields.deadline_confidence is DeadlineConfidence.MEDIUM
    assert fields.strong_reminder_eligible is True


def test_ambiguous_deadline_never_has_high_confidence() -> None:
    fields = extract_fields(document("请于本月底前提交"), NOW)

    assert fields.deadline_at == datetime(2026, 9, 30, tzinfo=SH_TZ)
    assert fields.deadline_confidence in {
        DeadlineConfidence.LOW,
        DeadlineConfidence.UNKNOWN,
    }
    assert fields.strong_reminder_eligible is False


def test_impossible_date_is_rejected() -> None:
    fields = extract_fields(document("报名截止：2026年2月30日"), NOW)

    assert fields.deadline_at is None
    assert fields.deadline_confidence is DeadlineConfidence.UNKNOWN
    assert fields.strong_reminder_eligible is False


def test_past_deadline_is_never_strongly_remindable() -> None:
    fields = extract_fields(document("报名截止：2026年9月1日"), NOW)

    assert fields.deadline_at == datetime(2026, 9, 1, tzinfo=SH_TZ)
    assert fields.deadline_confidence is DeadlineConfidence.LOW
    assert fields.strong_reminder_eligible is False


def test_date_only_deadline_uses_low_confidence_without_invented_time() -> None:
    fields = extract_fields(document("报名截止：2026年10月11日"), NOW)
    assert fields.deadline_at == datetime(2026, 10, 11, tzinfo=SH_TZ)
    assert fields.deadline_confidence is DeadlineConfidence.LOW
    assert fields.strong_reminder_eligible is False


def test_month_day_only_deadline_is_also_low_confidence() -> None:
    fields = extract_fields(document("报名截止：10月11日"), NOW)
    assert fields.deadline_at == datetime(2026, 10, 11, tzinfo=SH_TZ)
    assert fields.deadline_confidence is DeadlineConfidence.LOW
    assert fields.strong_reminder_eligible is False
