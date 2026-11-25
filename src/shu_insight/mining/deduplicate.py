from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from datetime import timedelta
from difflib import SequenceMatcher
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from shu_insight.domain.models import NoticeRecord


@dataclass(frozen=True, slots=True)
class CandidatePair:
    left: NoticeRecord
    right: NoticeRecord
    reason: str


@dataclass(frozen=True, slots=True)
class SimilarityEvidence:
    left_id: str
    right_id: str
    score: float
    title_similarity: float
    simhash_similarity: float
    structured_overlap: float
    exact_match: bool = False
    disposition: str = "candidate"


@dataclass(frozen=True, slots=True)
class DuplicateCluster:
    id: str
    members: tuple[NoticeRecord, ...]
    evidence: tuple[SimilarityEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class ClusterAssignments:
    clusters: tuple[DuplicateCluster, ...]
    possible_duplicates: tuple[SimilarityEvidence, ...] = ()

    def __len__(self) -> int:
        return len(self.clusters)

    def __iter__(self):
        return iter(self.clusters)

    def __getitem__(self, index: int) -> DuplicateCluster:
        return self.clusters[index]


def _normalized_title(title: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", title).casefold()


def _title_similarity(left: str, right: str) -> float:
    if _normalized_title(left) == _normalized_title(right):
        return 1.0
    matrix = TfidfVectorizer(analyzer="char", ngram_range=(1, 3)).fit_transform([left, right])
    tfidf_score = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
    sequence_score = SequenceMatcher(
        None, _normalized_title(left), _normalized_title(right)
    ).ratio()
    return max(tfidf_score, sequence_score)


def _simhash_similarity(left: int, right: int) -> float:
    different_bits = (left ^ right).bit_count()
    return 1.0 - different_bits / 63.0


def _jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float | None:
    union = set(left) | set(right)
    if not union:
        return None
    return len(set(left) & set(right)) / len(union)


def _structured_overlap(left: NoticeRecord, right: NoticeRecord) -> float:
    parts = [
        score
        for score in (
            _jaccard(left.categories, right.categories),
            _jaccard(left.topic_tags, right.topic_tags),
            _jaccard(left.majors, right.majors),
        )
        if score is not None
    ]
    if left.published_at and right.published_at:
        day_gap = abs((left.published_at - right.published_at).total_seconds()) / 86_400
        parts.append(max(0.0, 1.0 - day_gap / 30.0))
    if left.location and right.location:
        parts.append(1.0 if left.location == right.location else 0.0)
    if left.department and right.department:
        parts.append(1.0 if left.department == right.department else 0.0)
    return sum(parts) / len(parts) if parts else 0.0


def _within_window(left: NoticeRecord, right: NoticeRecord) -> bool:
    if not left.published_at or not right.published_at:
        return True
    return abs(left.published_at - right.published_at) <= timedelta(days=90)


def candidate_pairs(notices: list[NoticeRecord]) -> tuple[CandidatePair, ...]:
    pairs: list[CandidatePair] = []
    for left, right in combinations(notices, 2):
        exact_url = left.canonical_url == right.canonical_url
        exact_title = _normalized_title(left.title) == _normalized_title(right.title)
        if exact_url or exact_title:
            pairs.append(CandidatePair(left, right, "exact_url" if exact_url else "exact_title"))
        elif _within_window(left, right):
            pairs.append(CandidatePair(left, right, "publication_window"))
    return tuple(pairs)


def duplicate_similarity(left: NoticeRecord, right: NoticeRecord) -> SimilarityEvidence:
    exact = left.canonical_url == right.canonical_url or (
        _normalized_title(left.title) == _normalized_title(right.title)
    )
    if exact:
        return SimilarityEvidence(
            left_id=left.id,
            right_id=right.id,
            score=1.0,
            title_similarity=1.0,
            simhash_similarity=_simhash_similarity(left.summary_simhash, right.summary_simhash),
            structured_overlap=_structured_overlap(left, right),
            exact_match=True,
        )

    title_score = _title_similarity(left.title, right.title)
    simhash_score = _simhash_similarity(left.summary_simhash, right.summary_simhash)
    structured_score = _structured_overlap(left, right)
    score = 0.55 * title_score + 0.25 * simhash_score + 0.20 * structured_score
    return SimilarityEvidence(
        left_id=left.id,
        right_id=right.id,
        score=score,
        title_similarity=title_score,
        simhash_similarity=simhash_score,
        structured_overlap=structured_score,
        exact_match=False,
    )


def _cluster_id(members: list[NoticeRecord]) -> str:
    payload = "\n".join(sorted(member.id for member in members)).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def assign_clusters(
    notices: list[NoticeRecord], threshold: float = 0.78
) -> ClusterAssignments:
    parent = {notice.id: notice.id for notice in notices}

    def find(notice_id: str) -> str:
        while parent[notice_id] != notice_id:
            parent[notice_id] = parent[parent[notice_id]]
            notice_id = parent[notice_id]
        return notice_id

    def union(left_id: str, right_id: str) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root != right_root:
            parent[right_root] = left_root

    merged_evidence: list[SimilarityEvidence] = []
    possible: list[SimilarityEvidence] = []
    for pair in candidate_pairs(notices):
        evidence = duplicate_similarity(pair.left, pair.right)
        if evidence.score >= threshold:
            union(pair.left.id, pair.right.id)
            merged_evidence.append(replace(evidence, disposition="duplicate"))
        elif evidence.score >= max(0.50, threshold - 0.35):
            possible.append(replace(evidence, disposition="possible_duplicate"))

    grouped: dict[str, list[NoticeRecord]] = {}
    for notice in notices:
        grouped.setdefault(find(notice.id), []).append(notice)

    clusters: list[DuplicateCluster] = []
    for members in grouped.values():
        member_ids = {member.id for member in members}
        evidence = tuple(
            item
            for item in merged_evidence
            if item.left_id in member_ids and item.right_id in member_ids
        )
        clusters.append(
            DuplicateCluster(
                id=_cluster_id(members),
                members=tuple(members),
                evidence=evidence,
            )
        )
    clusters.sort(key=lambda cluster: min(member.id for member in cluster.members))
    return ClusterAssignments(tuple(clusters), tuple(possible))
