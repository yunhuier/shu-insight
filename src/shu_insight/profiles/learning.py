from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from jieba.analyse import extract_tags
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from shu_insight.db.repositories import ArticleReadRepository
from shu_insight.domain.models import (
    NoticeRecord,
    ParseState,
    Persona,
)
from shu_insight.processing.classify import Taxonomy, classify_notice
from shu_insight.profiles.service import (
    PersonaValidationError,
    majors_for,
    validate_persona,
)

LEARN_PERSONA_ID = "learned-search-cluster"
LEARN_LABEL = "聚类画像"
LEARN_UNAVAILABLE_SUFFIX = "（阅读篇数不足，暂不可用）"
MINIMUM_ARTICLES = 10
MAX_CLUSTERS = 4

_EDUCATION_LEVELS = frozenset({"本科生", "硕士生", "博士生"})
_LEVEL_ALIASES = {
    "本科": "本科生",
    "硕士": "硕士生",
    "博士": "博士生",
}
_GRADE_LEVELS = {
    "本科生": frozenset({"大一", "大二", "大三", "大四"}),
    "硕士生": frozenset({"研一", "研二", "研三"}),
    "博士生": frozenset({"博一", "博二", "博三", "博四", "博五"}),
}


@dataclass(frozen=True, slots=True)
class LearnedPersona:
    """The persona fitted from the local search history, plus how it was fitted."""

    persona: Persona
    article_count: int
    cluster_count: int
    cluster_index: int
    cluster_size: int
    representative_terms: tuple[str, ...]
    silhouette: float | None = None


def _recency_weight(moment: datetime, now: datetime) -> float:
    age_days = max((now - moment).total_seconds() / 86400.0, 0.0)
    return max(0.5 ** (age_days / 30.0), 0.2)


@dataclass(frozen=True, slots=True)
class _ReadArticle:
    notice: NoticeRecord
    created_at: datetime


class ArticleReadLearner:
    """Cluster the content of distinct articles the user explicitly opens."""

    def __init__(
        self,
        repository: ArticleReadRepository,
        taxonomy: Taxonomy,
        *,
        notice_source: Callable[[], Sequence[NoticeRecord]],
        minimum_articles: int = MINIMUM_ARTICLES,
        max_clusters: int = MAX_CLUSTERS,
        random_state: int = 42,
    ) -> None:
        self.repository = repository
        self.taxonomy = taxonomy
        self.notice_source = notice_source
        self.minimum_articles = minimum_articles
        self.max_clusters = max_clusters
        self.random_state = random_state

    def read_count(self) -> int:
        return self.repository.count()

    def record_read(self, notice: NoticeRecord) -> bool:
        if not any(item.id == notice.id for item in self.notice_source()):
            return False
        return self.repository.record(notice.id)

    def learn(self, *, now: datetime | None = None) -> LearnedPersona | None:
        entries = self.repository.entries(limit=200)
        if len(entries) < self.minimum_articles:
            return None
        moment = now or datetime.now(UTC)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        notices = {notice.id: notice for notice in self.notice_source()}
        usable = [_ReadArticle(notices[entry.notice_id], entry.created_at)
                  for entry in entries if entry.notice_id in notices]
        if len(usable) < self.minimum_articles:
            return None

        weights = np.array([_recency_weight(entry.created_at, moment) for entry in usable])
        matrix, vocabulary = self._vectorize([self._document(article) for article in usable])
        if matrix is None or vocabulary is None:
            return None

        cluster_count = min(self.max_clusters, max(2, len(usable) // 4), len(usable) - 1)
        label_vector, silhouette = self._fit(matrix, weights, cluster_count)
        active = int(label_vector[-1])
        members = [entry for entry, label in zip(usable, label_vector, strict=True)
                   if int(label) == active]
        representative = self._representative_terms(members, moment)
        if not representative:
            return None

        persona = self._build_persona(
            representative,
            members,
            usable,
            moment,
        )
        if persona is None:
            return None
        return LearnedPersona(
            persona=persona,
            article_count=len(usable),
            cluster_count=len(set(int(label) for label in label_vector)),
            cluster_index=active + 1,
            cluster_size=len(members),
            representative_terms=representative,
            silhouette=silhouette,
        )

    # -- feature space -------------------------------------------------
    @staticmethod
    def _document(article: _ReadArticle) -> str:
        notice = article.notice
        if notice.parse_state is ParseState.LISTING_ONLY:
            return notice.title
        return " ".join((notice.title, notice.title, notice.summary,
                         *notice.topic_tags, *notice.categories))

    def _vectorize(
        self, documents: list[str]
    ) -> tuple[Any | None, Any | None]:
        try:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 3),
                sublinear_tf=True,
                lowercase=False,
            )
            matrix = vectorizer.fit_transform(documents)
        except ValueError:
            return None, None
        if matrix.shape[1] == 0:
            return None, None
        return matrix, vectorizer

    def _fit(
        self, matrix: Any, weights: np.ndarray, cluster_count: int
    ) -> tuple[np.ndarray, float | None]:
        try:
            model = KMeans(
                n_clusters=cluster_count,
                random_state=self.random_state,
                n_init=10,
            )
            labels = model.fit_predict(matrix, sample_weight=weights)
        except ValueError:
            labels = np.zeros(matrix.shape[0], dtype=int)
            return labels, None
        score: float | None
        try:
            if len(set(int(label) for label in labels)) < 2:
                score = None
            else:
                score = float(silhouette_score(matrix, labels))
        except ValueError:
            score = None
        return labels, score

    def _representative_terms(
        self, members: Sequence[_ReadArticle], now: datetime, limit: int = 6
    ) -> tuple[str, ...]:
        counter: Counter[str] = Counter()
        for article in members:
            notice = article.notice
            weight = _recency_weight(article.created_at, now)
            summary = "" if notice.parse_state is ParseState.LISTING_ONLY else notice.summary
            classification = classify_notice(notice.title, summary, self.taxonomy)
            terms = set(match.term.casefold() for match in classification.matches)
            terms.update(classification.tags)
            if notice.parse_state is not ParseState.LISTING_ONLY:
                terms.update((*notice.topic_tags, *notice.categories))
            for term in terms:
                if len(term) >= 2:
                    counter[term] += weight
        if not counter:
            text = "\n".join(self._document(article) for article in members)
            counter.update(extract_tags(text, topK=limit))
        ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        return tuple(term for term, _ in ordered[:limit])

    # -- persona derivation -------------------------------------------
    def _build_persona(
        self,
        terms: tuple[str, ...],
        members: Sequence[_ReadArticle],
        articles: Sequence[_ReadArticle],
        moment: datetime,
    ) -> Persona | None:
        goal_scores = self._label_scores(self.taxonomy.tags, terms)
        category_scores = self._label_scores(self.taxonomy.categories, terms)
        if not goal_scores:
            goal_scores = Counter(
                {tag: 1.0 for article in members for tag in article.notice.topic_tags[:2]
                 if article.notice.parse_state is not ParseState.LISTING_ONLY}
            )
        if not category_scores:
            category_scores = Counter(
                {name: 1.0 for article in members for name in article.notice.categories[:2]
                 if article.notice.parse_state is not ParseState.LISTING_ONLY}
            )

        goals = tuple(label for label, _ in goal_scores.most_common(2))
        interest_tags = tuple(
            dict.fromkeys(
                [label for label, _ in category_scores.most_common(3)]
                + [label for label, _ in goal_scores.most_common(3)]
            )
        )[:8]
        custom_keywords = tuple(
            term for term in terms if term not in {tag.casefold() for tag in interest_tags}
        )[:5]

        education, grade, college, major = self._infer_audience(articles, moment)
        if major and major not in majors_for(education, college):
            major = "不限专业"
        label = "我的阅读兴趣·" + ("、".join(goals) if goals else "综合")
        persona = Persona(
            id=LEARN_PERSONA_ID,
            label=label,
            education_level=education,
            grade=grade,
            college=college,
            major=major,
            goals=goals,
            interest_tags=interest_tags,
            custom_keywords=custom_keywords,
        )
        try:
            return validate_persona(persona)
        except PersonaValidationError:
            try:
                return validate_persona(
                    Persona(
                        id=LEARN_PERSONA_ID,
                        label=label,
                        goals=goals,
                        interest_tags=interest_tags,
                        custom_keywords=custom_keywords[:5],
                    )
                )
            except PersonaValidationError:
                return None

    @staticmethod
    def _label_scores(
        table: dict[str, tuple[str, ...]], terms: Sequence[str]
    ) -> Counter[str]:
        scores: Counter[str] = Counter()
        for label, keywords in table.items():
            for term in terms:
                if any(keyword.casefold() in term or term in keyword.casefold()
                       for keyword in keywords):
                    scores[label] += 1.0
        return scores

    def _infer_audience(
        self, articles: Sequence[_ReadArticle], moment: datetime,
    ) -> tuple[str, str, str, str]:
        """Vote over every opened article so a small active cluster still speaks."""
        votes = [
            (article.notice, _recency_weight(article.created_at, moment))
            for article in articles
            if article.notice.parse_state is not ParseState.LISTING_ONLY
        ]
        if len(votes) < 2:
            return "不限", "不限", "", "不限专业"

        def vote(
            values: Callable[[NoticeRecord], Sequence[str]],
            allowed: frozenset[str] | None = None,
        ) -> tuple[str, float]:
            counter: Counter[str] = Counter()
            total = 0.0
            for notice, weight in votes:
                total += weight
                for value in values(notice):
                    cleaned = _LEVEL_ALIASES.get(value.strip(), value.strip())
                    if cleaned and (allowed is None or cleaned in allowed):
                        counter[cleaned] += weight
            if not counter or total <= 0:
                return "", 0.0
            value, weight = counter.most_common(1)[0]
            return value, weight / total

        education, education_share = vote(
            lambda notice: notice.education_levels, _EDUCATION_LEVELS,
        )
        if education_share < 0.4:
            education = ""
        grade = college = major = ""
        if education:
            grade, grade_share = vote(
                lambda notice: notice.grades, _GRADE_LEVELS.get(education, frozenset()),
            )
            if grade_share < 0.3:
                grade = ""
        college, college_share = vote(lambda notice: notice.colleges)
        if college_share < 0.3:
            college = ""
        major, major_share = vote(
            lambda notice: tuple(self._canonical_major(value) for value in notice.majors),
        )
        if major_share < 0.3:
            major = ""
        return education or "不限", grade or "不限", college, major or "不限专业"

    def _canonical_major(self, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            return ""
        folded = cleaned.casefold()
        for label, keywords in self.taxonomy.majors.items():
            if folded == label.casefold():
                return label
            if any(keyword.casefold() in folded or folded in keyword.casefold()
                   for keyword in keywords):
                return label
        return cleaned

    @staticmethod
    def _mode(values: Any, minimum: int = 2) -> str:
        counter = Counter(value.strip() for value in values if value and value.strip())
        if not counter:
            return ""
        value, count = counter.most_common(1)[0]
        return value if count >= minimum else ""
