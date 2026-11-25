from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from shu_insight.domain.models import (
    DeadlineConfidence,
    MatchTier,
    NoticeRecord,
    ParseState,
    Persona,
    ProofStatus,
    Recommendation,
    ScoreBreakdown,
)
from shu_insight.processing.audience import explicit_audience_scope
from shu_insight.processing.dates import SHANGHAI_TZ
from shu_insight.ranking.expand import ExpandedQuery
from shu_insight.ranking.relevance import ExactBackend, RelevanceResult


@dataclass(frozen=True, slots=True)
class ScoreInputs:
    relevance: float
    audience: float
    urgency: float
    authority: float
    freshness: float


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    relevance: float
    audience: float
    urgency: float
    authority: float
    freshness: float


@dataclass(frozen=True, slots=True)
class RankingWeights:
    profile: ScoreWeights
    no_profile: ScoreWeights


@dataclass(frozen=True, slots=True)
class Timeline:
    dated: list[NoticeRecord]
    unknown: list[NoticeRecord]
    expired: list[NoticeRecord]

    @property
    def all_notices(self) -> list[NoticeRecord]:
        return [*self.dated, *self.expired, *self.unknown]


DEFAULT_WEIGHTS = RankingWeights(
    profile=ScoreWeights(0.40, 0.25, 0.20, 0.10, 0.05),
    no_profile=ScoreWeights(0.55, 0.00, 0.25, 0.10, 0.10),
)

# These broad words in a summary alone are weak evidence of personal relevance.
GENERIC_SUMMARY_TERMS = frozenset({"实验室", "科研", "学术", "论文"})
TIER_ORDER = {MatchTier.DIRECT: 0, MatchTier.POSSIBLE: 1, MatchTier.OTHER: 2}


def _score(
    parts: ScoreInputs, weights: ScoreWeights, *, include_audience: bool = True,
) -> float:
    """Weighted sum; a dropped audience term re-normalises the others.

    The accumulation order mirrors the handover build so rankings without a
    persona stay bit-identical.
    """
    weighted = (
        weights.relevance * parts.relevance
        + (weights.audience * parts.audience if include_audience else 0.0)
        + weights.urgency * parts.urgency
        + weights.authority * parts.authority
        + weights.freshness * parts.freshness
    )
    if include_audience:
        return min(1.0, max(0.0, weighted))
    total_weight = (
        weights.relevance + weights.urgency + weights.authority + weights.freshness
    )
    if total_weight <= 0:
        return 0.0
    return min(1.0, max(0.0, weighted / total_weight))


def score_profile(parts: ScoreInputs) -> float:
    return _score(parts, DEFAULT_WEIGHTS.profile)


def score_no_profile(parts: ScoreInputs) -> float:
    return _score(parts, DEFAULT_WEIGHTS.no_profile)


def load_ranking_weights(path: Path) -> RankingWeights:
    with path.open("rb") as config_file:
        document = tomllib.load(config_file)

    def read(section: str) -> ScoreWeights:
        raw = document[section]
        weights = ScoreWeights(
            relevance=float(raw["relevance"]),
            audience=float(raw.get("audience", 0.0)),
            urgency=float(raw["urgency"]),
            authority=float(raw["authority"]),
            freshness=float(raw["freshness"]),
        )
        total = (
            weights.relevance
            + weights.audience
            + weights.urgency
            + weights.authority
            + weights.freshness
        )
        if not math.isclose(total, 1.0):
            raise ValueError(f"ranking weights in {section!r} must sum to 1")
        return weights

    return RankingWeights(profile=read("profile"), no_profile=read("no_profile"))


def can_strong_remind(notice: NoticeRecord, now: datetime) -> bool:
    return bool(
        notice.parse_state is not ParseState.LISTING_ONLY
        and notice.deadline_at
        and notice.deadline_at > now
        and notice.deadline_confidence
        in {DeadlineConfidence.HIGH, DeadlineConfidence.MEDIUM}
    )


def _deadline_end(notice: NoticeRecord) -> datetime | None:
    if notice.deadline_at is None:
        return None
    if notice.deadline_confidence is DeadlineConfidence.LOW:
        local_day = notice.deadline_at.astimezone(SHANGHAI_TZ).date()
        return datetime.combine(local_day + timedelta(days=1), datetime.min.time(),
                                SHANGHAI_TZ)
    return notice.deadline_at


def _freshness(notice: NoticeRecord, now: datetime) -> float:
    if notice.published_at is None or notice.published_at > now:
        return 0.0
    age_days = (now - notice.published_at).total_seconds() / 86_400
    if age_days <= 7:
        return 1.0
    if age_days <= 30:
        return 0.7
    if age_days <= 90:
        return 0.4
    return 0.1


def _urgency(notice: NoticeRecord, now: datetime) -> float:
    if notice.parse_state is ParseState.LISTING_ONLY:
        return 0.0
    deadline_end = _deadline_end(notice)
    if deadline_end is None or deadline_end <= now:
        return 0.0
    days = (deadline_end - now).total_seconds() / 86_400
    if days <= 3:
        value = 1.0
    elif days <= 7:
        value = 0.8
    elif days <= 30:
        value = 0.5
    else:
        value = 0.2
    if notice.deadline_confidence is DeadlineConfidence.LOW:
        return value * 0.25
    if notice.deadline_confidence is DeadlineConfidence.UNKNOWN:
        return 0.0
    return value


def audience_evidence(notice: NoticeRecord, persona: Persona) -> tuple[tuple[str, ...], bool]:
    """Return explicit audience matches and conflicts; absent fields are unknown."""
    _, reasons, conflict = _structural_audience(notice, persona)
    return reasons, conflict


def _scope_evidence(notice: NoticeRecord, persona: Persona) -> tuple[tuple[str, ...], bool]:
    scope = explicit_audience_scope(
        notice.title,
        notice.summary if notice.parse_state is ParseState.PARSED and notice.proof_status
        in {ProofStatus.VERIFIED_LIVE, ProofStatus.VERIFIED_SNAPSHOT} else "",
    )
    level = persona.education_level
    known_level = level not in {"", "不限"}
    student_level = level in {"本科生", "研究生", "硕士生", "博士生"}
    if scope.teacher_only and student_level:
        return (), True
    # Graduation years vary by degree/program; only explicit early undergraduate
    # years provide a conservative conflict, without claiming graduation = 大四.
    if scope.graduating_students and level == "本科生" and persona.grade in {"大一", "大二"}:
        return (), True
    if scope.undergraduate_grades:
        if known_level and level != "本科生":
            return (), True
        if persona.grade not in {"", "不限"}:
            if persona.grade not in scope.undergraduate_grades:
                return (), True
            return (f"适用于{persona.grade}",), False
    return (), False


# Notices and personas name the same level in different ways (本科/本科生), so
# compare canonical forms before treating a different value as a conflict.
_EDUCATION_ALIASES = {
    "本科": "本科生",
    "本科生": "本科生",
    "研究生": "研究生",
    "硕士": "硕士生",
    "硕士生": "硕士生",
    "博士": "博士生",
    "博士生": "博士生",
}
_GRADUATE_LEVELS = frozenset({"研究生", "硕士生", "博士生"})
_GENERIC_AUDIENCE = frozenset(
    {"", "不限", "不限专业", "全校", "全体学生", "所有学生", "全校学生"}
)
AUDIENCE_STRUCTURAL_SHARE = 0.7
AUDIENCE_THEME_SHARE = 0.3
THEME_TERM_CAP = 3


def _canonical_audience(value: str) -> str:
    cleaned = value.strip()
    return _EDUCATION_ALIASES.get(cleaned, cleaned)


def _audience_match(allowed: Sequence[str], selected: str) -> bool | None:
    """True/False when the notice states an audience, None when it says nothing."""
    cleaned = selected.strip()
    if cleaned in _GENERIC_AUDIENCE:
        return None
    values = [
        _canonical_audience(value)
        for value in allowed
        if value.strip() and value.strip() not in _GENERIC_AUDIENCE
    ]
    if not values:
        return None
    target = _canonical_audience(cleaned)
    for value in values:
        if value == target:
            return True
        if target in _GRADUATE_LEVELS and value in _GRADUATE_LEVELS and "研究生" in {target, value}:
            return True
        if len(value) >= 2 and len(target) >= 2 and (value in target or target in value):
            return True
    return False


def _structural_audience(
    notice: NoticeRecord, persona: Persona,
) -> tuple[float | None, tuple[str, ...], bool]:
    """Score the explicit audience fields; None means the notice is silent."""
    scope_reasons, scope_conflict = _scope_evidence(notice, persona)
    if scope_conflict:
        return 0.0, tuple(scope_reasons), True
    reasons: list[str] = list(scope_reasons)
    values: list[float] = [1.0] if scope_reasons else []
    conflict = False
    if notice.parse_state is not ParseState.LISTING_ONLY:
        comparisons = (
            (notice.education_levels, persona.education_level),
            (notice.grades, persona.grade),
            (notice.colleges, persona.college),
            (notice.majors, persona.major),
        )
        for allowed, selected in comparisons:
            matched = _audience_match(allowed, selected)
            if matched is None:
                continue
            values.append(1.0 if matched else 0.0)
            if matched:
                reasons.append(f"适用于{selected}")
            else:
                conflict = True
    if not values:
        return None, tuple(dict.fromkeys(reasons)), conflict
    return sum(values) / len(values), tuple(dict.fromkeys(reasons)), conflict


def _theme_audience(notice: NoticeRecord, theme_terms: Sequence[str]) -> float | None:
    """Coverage of the persona's themes by the notice's own topics."""
    terms = tuple(dict.fromkeys(term.strip().casefold() for term in theme_terms
                                if term and term.strip()))
    if not terms:
        return None
    pool = {
        value.casefold()
        for value in (*notice.topic_tags, *notice.categories)
        if value and value.strip()
    }
    if pool:
        hits = sum(
            1
            for term in terms
            if any(term == tag or term in tag or tag in term for tag in pool)
        )
    else:
        text = notice.title.casefold()
        if notice.parse_state is ParseState.PARSED:
            text = f"{text} {notice.summary.casefold()}"
        hits = sum(1 for term in terms if term in text)
    return min(1.0, hits / min(len(terms), THEME_TERM_CAP))


@dataclass(frozen=True, slots=True)
class AudienceParts:
    """The persona dimension split into an explicit-audience and a theme part."""

    score: float
    structural: float
    theme: float
    structural_known: bool
    theme_known: bool
    known: bool
    reasons: tuple[str, ...]
    conflict: bool


def _audience_parts(
    notice: NoticeRecord, persona: Persona, theme_terms: Sequence[str],
) -> AudienceParts:
    structural, reasons, conflict = _structural_audience(notice, persona)
    if conflict:
        return AudienceParts(0.0, 0.0, 0.0, structural is not None, False,
                             True, reasons, True)
    theme = _theme_audience(notice, theme_terms)
    parts: list[tuple[float, float]] = []
    if structural is not None:
        parts.append((structural, AUDIENCE_STRUCTURAL_SHARE))
    if theme is not None:
        parts.append((theme, AUDIENCE_THEME_SHARE))
    if not parts:
        return AudienceParts(0.5, 0.0, 0.0, False, False, False, reasons, False)
    weight = sum(share for _, share in parts)
    value = sum(score * share for score, share in parts) / weight
    return AudienceParts(
        value,
        structural if structural is not None else 0.0,
        theme if theme is not None else 0.0,
        structural is not None,
        theme is not None,
        True,
        reasons,
        False,
    )


def _audience(notice: NoticeRecord, persona: Persona) -> float:
    structural, _, _ = _structural_audience(notice, persona)
    return 0.5 if structural is None else structural


def _keyword_evidence(
    notice: NoticeRecord, relevance: RelevanceResult,
) -> tuple[frozenset[str], MatchTier]:
    # Recheck persisted fields: backend/tag matches must not invent textual proof.
    query = ExpandedQuery(
        original_terms=tuple(sorted(relevance.matched_terms)),
        suggested_terms=(), enabled_terms=(), evidence=(),
    )
    backend = ExactBackend()
    title_terms = backend.score(query, notice.title).matched_terms
    if notice.parse_state is ParseState.LISTING_ONLY:
        return title_terms, MatchTier.POSSIBLE if title_terms else MatchTier.OTHER
    if notice.parse_state is not ParseState.PARSED:
        return frozenset(), MatchTier.OTHER
    summary_terms = backend.score(query, notice.summary).matched_terms
    matched_terms = title_terms | summary_terms
    if title_terms or summary_terms - GENERIC_SUMMARY_TERMS:
        return matched_terms, MatchTier.DIRECT
    return matched_terms, MatchTier.POSSIBLE if summary_terms else MatchTier.OTHER


def _reasons(
    notice: NoticeRecord,
    matched_terms: frozenset[str],
    audience_reasons: tuple[str, ...],
    now: datetime,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if matched_terms:
        reasons.append(f"匹配关键词：{'、'.join(sorted(matched_terms))}")
    reasons.extend(audience_reasons)
    if can_strong_remind(notice, now):
        days = math.ceil((notice.deadline_at - now).total_seconds() / 86_400)
        reasons.append(f"距离截止时间 {days} 天")
    return tuple(reasons)


def _stale(notice: NoticeRecord, now: datetime) -> bool:
    deadline_end = (
        _deadline_end(notice) if notice.parse_state is not ParseState.LISTING_ONLY else None
    )
    return bool(
        (deadline_end is not None and deadline_end <= now)
        or (notice.published_at is not None and now - notice.published_at > timedelta(days=90))
    )


def rank_notices(
    notices: list[NoticeRecord],
    relevance_by_notice_id: Mapping[str, RelevanceResult],
    authority_by_source_id: Mapping[str, float],
    now: datetime,
    *,
    persona: Persona | None,
    weights: RankingWeights = DEFAULT_WEIGHTS,
    theme_terms: Sequence[str] = (),
) -> list[Recommendation]:
    recommendations: list[Recommendation] = []
    for notice in notices:
        relevance = relevance_by_notice_id.get(
            notice.id, RelevanceResult(0.0, frozenset(), "none")
        )
        if persona is None:
            audience = AudienceParts(0.0, 0.0, 0.0, False, False, False, (), False)
            selected_weights = weights.no_profile
        else:
            audience = _audience_parts(notice, persona, theme_terms)
            selected_weights = weights.profile
        matched_terms, match_tier = _keyword_evidence(notice, relevance)
        if audience.conflict:
            match_tier = MatchTier.OTHER
        elif audience.reasons and match_tier is MatchTier.OTHER:
            match_tier = MatchTier.POSSIBLE
        inputs = ScoreInputs(
            relevance=relevance.score,
            audience=audience.score,
            urgency=_urgency(notice, now),
            authority=min(1.0, max(0.0, authority_by_source_id.get(notice.source_id, 0.0))),
            freshness=_freshness(notice, now),
        )
        total = _score(
            inputs,
            selected_weights,
            include_audience=audience.known and selected_weights.audience > 0,
        )
        recommendations.append(
            Recommendation(
                notice=notice,
                score=ScoreBreakdown(
                    relevance=inputs.relevance,
                    audience=inputs.audience,
                    urgency=inputs.urgency,
                    authority=inputs.authority,
                    freshness=inputs.freshness,
                    total=total,
                    audience_known=audience.known,
                    audience_structural=audience.structural,
                    audience_theme=audience.theme,
                    audience_structural_known=audience.structural_known,
                    audience_theme_known=audience.theme_known,
                ),
                reasons=_reasons(notice, matched_terms, audience.reasons, now),
                strong_reminder_eligible=can_strong_remind(notice, now),
                match_tier=match_tier,
            )
        )

    minimum_date = datetime.min.replace(tzinfo=now.tzinfo)
    recommendations.sort(
        key=lambda item: (
            TIER_ORDER[item.match_tier],
            _stale(item.notice, now),
            -item.score.total,
            -(item.notice.published_at or minimum_date).timestamp(),
            item.notice.id,
        )
    )
    return recommendations


def latest_timeline(notices: list[NoticeRecord]) -> Timeline:
    dated = sorted(
        (notice for notice in notices if notice.published_at is not None),
        key=lambda notice: (notice.published_at, notice.id),
        reverse=True,
    )
    unknown = sorted(
        (notice for notice in notices if notice.published_at is None),
        key=lambda notice: notice.id,
    )
    return Timeline(dated=dated, unknown=unknown, expired=[])


def deadline_timeline(notices: list[NoticeRecord], now: datetime) -> Timeline:
    dated = sorted(
        (notice for notice in notices if (end := _deadline_end(notice)) and end > now),
        key=lambda notice: (notice.deadline_at, notice.id),
    )
    expired = sorted(
        (notice for notice in notices if (end := _deadline_end(notice)) and end <= now),
        key=lambda notice: (notice.deadline_at, notice.id),
        reverse=True,
    )
    unknown = sorted(
        (notice for notice in notices if notice.deadline_at is None),
        key=lambda notice: notice.id,
    )
    return Timeline(dated=dated, unknown=unknown, expired=expired)
