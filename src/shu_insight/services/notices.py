from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from shu_insight.compliance.detector import redact_public_title
from shu_insight.db.repositories import NoticeRepository
from shu_insight.domain.models import NoticeRecord, ParseState, Persona, Recommendation
from shu_insight.processing.classify import Taxonomy
from shu_insight.processing.public_preview import safe_stored_summary
from shu_insight.ranking.expand import ExpandedQuery, Query, expand_terms
from shu_insight.ranking.relevance import ExactBackend
from shu_insight.ranking.score import (
    DEFAULT_WEIGHTS,
    RankingWeights,
    Timeline,
    deadline_timeline,
    latest_timeline,
    rank_notices,
)


def _public_notice(notice: NoticeRecord) -> NoticeRecord:
    return replace(notice, title=redact_public_title(notice.title),
                   summary=safe_stored_summary(notice.summary))


def _public_evidence(value: Any) -> Any:
    """Copy display evidence while leaving proof identifiers and storage intact."""
    if isinstance(value, NoticeRecord):
        return _public_notice(value)
    if isinstance(value, list):
        return [_public_evidence(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_public_evidence(item) for item in value)
    if isinstance(value, dict):
        projected = {key: _public_evidence(item) for key, item in value.items()}
        for field, sanitizer in (("title", redact_public_title), ("summary", safe_stored_summary)):
            if isinstance(projected.get(field), str):
                projected[field] = sanitizer(projected[field])
            if value.get("field") == field:
                for side in ("before", "after"):
                    if isinstance(projected.get(side), str):
                        projected[side] = sanitizer(projected[side])
        return projected
    return value


@dataclass(frozen=True, slots=True)
class NoticeQuery:
    terms: tuple[str, ...] = ()
    persona: Persona | None = None
    minimum_relevance: float = 0.0
    limit: int | None = None


class NoticeService:
    def __init__(
        self,
        repository: NoticeRepository,
        *,
        authority_by_source_id: Mapping[str, float],
        ranking_weights: RankingWeights = DEFAULT_WEIGHTS,
        taxonomy: Taxonomy | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.authority_by_source_id = dict(authority_by_source_id)
        self.ranking_weights = ranking_weights
        self.taxonomy = taxonomy
        self._now = now or (lambda: datetime.now(UTC))

    def _theme_terms(self, persona: Persona) -> tuple[str, ...]:
        """Expand the persona's themes so notices tagged with sibling terms still match."""
        if self.taxonomy is None:
            return tuple(dict.fromkeys((*persona.goals, *persona.interest_tags)))
        expanded = expand_terms(
            Query(
                tags=persona.interest_tags or persona.goals,
                keywords=persona.custom_keywords,
            ),
            self.taxonomy,
        )
        return tuple(
            dict.fromkeys((*persona.goals, *persona.interest_tags, *expanded.all_terms))
        )

    def recommend(self, query: NoticeQuery) -> list[Recommendation]:
        notices = self.all()
        expanded = ExpandedQuery(
            original_terms=tuple(
                dict.fromkeys(term.strip() for term in query.terms if term.strip())
            ),
            suggested_terms=(),
            enabled_terms=(),
            evidence=(),
        )
        backend = ExactBackend()
        relevance = {
            notice.id: backend.score(
                expanded,
                notice.title if notice.parse_state is ParseState.LISTING_ONLY
                else f"{notice.title}\n{notice.summary}",
            )
            for notice in notices
        }
        matching = [
            notice
            for notice in notices
            if query.minimum_relevance <= 0
            or relevance[notice.id].score > query.minimum_relevance
        ]
        ranked = rank_notices(
            matching,
            relevance,
            self.authority_by_source_id,
            self._now(),
            persona=query.persona,
            weights=self.ranking_weights,
            theme_terms=(
                self._theme_terms(query.persona) if query.persona is not None else ()
            ),
        )
        return ranked[: query.limit] if query.limit is not None else ranked

    def latest(self) -> list[NoticeRecord]:
        return latest_timeline(self.all()).all_notices

    def upcoming(self) -> Timeline:
        return deadline_timeline(self.all(), self._now())

    def all(self) -> list[NoticeRecord]:
        return [_public_notice(notice) for notice in self.repository.list_browseable()]

    def detail(self, notice_id: str) -> NoticeRecord | None:
        return next(
            (_public_notice(notice) for notice in self.repository.list_verified()
             if notice.id == notice_id),
            None,
        )

    def detail_evidence(self, notice_id: str) -> dict:
        return _public_evidence(self.repository.detail_evidence_verified(notice_id))
