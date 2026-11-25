from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar
from zoneinfo import ZoneInfo

from shu_insight.domain.models import DeadlineConfidence
from shu_insight.processing.normalize import NormalizedDocument

T = TypeVar("T")
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")

DATE_PATTERN = re.compile(
    r"(?:(?P<year>20\d{2})\s*[-年/.]\s*)?"
    r"(?P<month>\d{1,2})\s*[-月/.]\s*"
    r"(?P<day>\d{1,2})\s*日?"
    r"(?:\s*(?P<hour>\d{1,2})[:：](?P<minute>\d{2}))?"
)
PREFIX_DEADLINE_CUES = ("截止", "报名时间", "提交时间", "申报时间")
SUFFIX_DEADLINE_CUES = ("前发送", "前提交")
AUDIENCE_TERMS = {
    "本科生": ("本科生", "本科", "大一", "大二", "大三", "大四"),
    "硕士生": ("硕士生", "硕士研究生"),
    "博士生": ("博士生", "博士研究生"),
}
GRADE_TERMS = ("大一", "大二", "大三", "大四", "研一", "研二", "研三", "博一", "博二")
COLLEGE_TERMS = (
    "计算机工程与科学学院",
    "通信与信息工程学院",
    "机电工程与自动化学院",
    "材料科学与工程学院",
    "经济学院",
    "管理学院",
    "钱伟长学院",
)
MAJOR_TERMS = (
    "计算机科学与技术",
    "人工智能",
    "智能科学与技术",
    "网络空间安全",
    "数据科学与大数据技术",
)
LOCATION_PATTERN = re.compile(r"(?:地点|地址)\s*[:：]\s*([^，。；;\n]{2,40})")
MATERIAL_PATTERN = re.compile(r"(?:提交材料|申报材料|报名材料)\s*[:：]\s*([^。；;\n]+)")
RELATIVE_MONTH_END_PATTERN = re.compile(r"本月底")


@dataclass(frozen=True, slots=True)
class Evidence(Generic[T]):
    value: T
    evidence_text: str
    start: int
    end: int
    confidence: float


@dataclass(frozen=True, slots=True)
class ExtractedFields:
    deadline_candidates: tuple[Evidence[datetime], ...] = ()
    selected_deadline: Evidence[datetime] | None = None
    deadline_confidence: DeadlineConfidence = DeadlineConfidence.UNKNOWN
    strong_reminder_eligible: bool = False
    locations: tuple[Evidence[str], ...] = ()
    education_levels: tuple[Evidence[str], ...] = ()
    grades: tuple[Evidence[str], ...] = ()
    colleges: tuple[Evidence[str], ...] = ()
    majors: tuple[Evidence[str], ...] = ()
    materials: tuple[Evidence[str], ...] = ()

    @property
    def deadline_at(self) -> datetime | None:
        return self.selected_deadline.value if self.selected_deadline else None

    def all_evidence(self) -> tuple[Evidence[object], ...]:
        groups = (
            self.deadline_candidates,
            self.locations,
            self.education_levels,
            self.grades,
            self.colleges,
            self.majors,
            self.materials,
        )
        return tuple(item for group in groups for item in group)  # type: ignore[misc]


def _evidence(
    value: T,
    text: str,
    start: int,
    end: int,
    confidence: float,
) -> Evidence[T]:
    return Evidence(
        value=value,
        evidence_text=text[start:end],
        start=start,
        end=end,
        confidence=confidence,
    )


def _cue_positions(text: str) -> tuple[tuple[int, int, bool], ...]:
    return tuple(
        (match.start(), match.end(), cue in PREFIX_DEADLINE_CUES)
        for cue in (*PREFIX_DEADLINE_CUES, *SUFFIX_DEADLINE_CUES)
        for match in re.finditer(re.escape(cue), text)
    )


def _distance_to_cue(
    start: int, end: int, cues: tuple[tuple[int, int, bool], ...]
) -> int:
    distances: list[int] = []
    for cue_start, cue_end, is_prefix in cues:
        if start >= cue_end:
            distances.append(start - cue_end)
        elif end <= cue_start:
            direction_penalty = 25 if is_prefix else 0
            distances.append(cue_start - end + direction_penalty)
        else:
            distances.append(0)
    return min(distances, default=10_000)


def _date_candidates(
    text: str, now: datetime
) -> tuple[
    list[tuple[Evidence[datetime], DeadlineConfidence, int]],
    tuple[tuple[int, int, bool], ...],
]:
    cues = _cue_positions(text)
    candidates: list[tuple[Evidence[datetime], DeadlineConfidence, int]] = []
    timezone = now.tzinfo or SHANGHAI_TZ
    for match in DATE_PATTERN.finditer(text):
        year_text = match.group("year")
        year = int(year_text) if year_text else now.year
        month = int(match.group("month"))
        day = int(match.group("day"))
        hour_text = match.group("hour")
        minute_text = match.group("minute")
        hour = int(hour_text) if hour_text else 0
        minute = int(minute_text) if minute_text else 0
        try:
            value = datetime(year, month, day, hour, minute, tzinfo=timezone)
        except ValueError:
            continue
        distance = _distance_to_cue(match.start(), match.end(), cues)
        has_time = hour_text is not None
        confidence = (DeadlineConfidence.HIGH if year_text else DeadlineConfidence.MEDIUM
                      ) if has_time else DeadlineConfidence.LOW
        numeric_confidence = (0.95 if year_text else 0.75) if has_time else 0.45
        candidates.append(
            (
                _evidence(
                    value,
                    text,
                    match.start(),
                    match.end(),
                    numeric_confidence,
                ),
                confidence,
                distance,
            )
        )
    return candidates, cues


def _relative_month_end(
    text: str, now: datetime, cues: tuple[tuple[int, int, bool], ...]
) -> tuple[Evidence[datetime], DeadlineConfidence, int] | None:
    match = RELATIVE_MONTH_END_PATTERN.search(text)
    if match is None:
        return None
    last_day = calendar.monthrange(now.year, now.month)[1]
    timezone = now.tzinfo or SHANGHAI_TZ
    value = datetime(now.year, now.month, last_day, tzinfo=timezone)
    return (
        _evidence(value, text, match.start(), match.end(), 0.35),
        DeadlineConfidence.LOW,
        _distance_to_cue(match.start(), match.end(), cues),
    )


def _term_evidence(
    text: str, canonical: str, terms: tuple[str, ...], confidence: float = 0.9
) -> Evidence[str] | None:
    matches = [
        match
        for term in sorted(terms, key=len, reverse=True)
        if (match := re.search(re.escape(term), text)) is not None
    ]
    if not matches:
        return None
    match = min(matches, key=lambda item: item.start())
    return _evidence(canonical, text, match.start(), match.end(), confidence)


def _all_fixed_terms(text: str, terms: tuple[str, ...]) -> tuple[Evidence[str], ...]:
    found: list[Evidence[str]] = []
    for term in terms:
        match = re.search(re.escape(term), text)
        if match is not None:
            found.append(_evidence(term, text, match.start(), match.end(), 0.9))
    return tuple(found)


def _locations(text: str) -> tuple[Evidence[str], ...]:
    return tuple(
        _evidence(match.group(1).strip(), text, match.start(1), match.end(1), 0.9)
        for match in LOCATION_PATTERN.finditer(text)
    )


def _materials(text: str) -> tuple[Evidence[str], ...]:
    results: list[Evidence[str]] = []
    for match in MATERIAL_PATTERN.finditer(text):
        cursor = match.start(1)
        for raw_part in re.split(r"[、,，]", match.group(1)):
            part = raw_part.strip()
            if not part:
                continue
            start = text.find(part, cursor, match.end(1))
            if start < 0:
                continue
            end = start + len(part)
            results.append(_evidence(part, text, start, end, 0.85))
            cursor = end
    return tuple(results)


def extract_fields(document: NormalizedDocument, now: datetime) -> ExtractedFields:
    """Extract reproducible candidate facts while preserving source evidence."""

    now = (now if now.tzinfo else now.replace(tzinfo=SHANGHAI_TZ)).astimezone(SHANGHAI_TZ)
    text = document.text
    candidate_rows, cues = _date_candidates(text, now)
    relative = _relative_month_end(text, now, cues)
    if relative is not None:
        candidate_rows.append(relative)

    selected_row = min(candidate_rows, key=lambda row: row[2]) if cues and candidate_rows else None
    if selected_row is not None and selected_row[2] > 50:
        selected_row = None
    selected = selected_row[0] if selected_row else None
    deadline_confidence = selected_row[1] if selected_row else DeadlineConfidence.UNKNOWN
    strong_reminder = bool(
        selected
        and selected.value > now
        and deadline_confidence in {DeadlineConfidence.HIGH, DeadlineConfidence.MEDIUM}
    )

    education_levels = tuple(
        evidence
        for canonical, terms in AUDIENCE_TERMS.items()
        if (evidence := _term_evidence(text, canonical, terms)) is not None
    )
    return ExtractedFields(
        deadline_candidates=tuple(row[0] for row in candidate_rows),
        selected_deadline=selected,
        deadline_confidence=deadline_confidence,
        strong_reminder_eligible=strong_reminder,
        locations=_locations(text),
        education_levels=education_levels,
        grades=_all_fixed_terms(text, GRADE_TERMS),
        colleges=_all_fixed_terms(text, COLLEGE_TERMS),
        majors=_all_fixed_terms(text, MAJOR_TERMS),
        materials=_materials(text),
    )
