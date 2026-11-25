from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum


class FetchState(StrEnum):
    SUCCESS = "success"
    NOT_MODIFIED = "not_modified"
    RATE_LIMITED = "rate_limited"
    FORBIDDEN = "forbidden"
    ROBOTS_DISALLOWED = "robots_disallowed"
    LOGIN_REQUIRED = "login_required"
    ERROR = "error"


class ParseState(StrEnum):
    PARSED = "parsed"
    LISTING_ONLY = "listing_only"
    BLOCKED = "blocked"
    STRUCTURE_CHANGED = "structure_changed"
    FAILED = "failed"


class ProofStatus(StrEnum):
    UNVERIFIED_LEGACY = "unverified_legacy"
    VERIFIED_SNAPSHOT = "verified_snapshot"
    VERIFIED_LIVE = "verified_live"
    VERIFIED_LISTING = "verified_listing"
    ATTACHMENT_EXCLUDED = "attachment_excluded"


@dataclass(frozen=True, slots=True)
class Verification:
    status: ProofStatus
    verified_at: datetime
    reference: str


class DeadlineConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class MatchTier(StrEnum):
    DIRECT = "direct"
    POSSIBLE = "possible"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class SourceConfig:
    id: str
    name: str
    base_url: str
    list_url: str
    allowed_path_prefixes: tuple[str, ...]
    authority_score: float
    minimum_interval_seconds: float
    reviewed_on: date
    pagination_selector: str | None = None
    pagination_url_pattern: str | None = None
    pagination_reviewed_on: date | None = None
    pagination_link_text: str | None = None


@dataclass(frozen=True, slots=True)
class DiscoveredNotice:
    source_id: str
    url: str
    title: str
    published_hint: str | None = None


@dataclass(frozen=True, slots=True)
class FetchedPage:
    source_id: str
    url: str
    final_url: str
    status_code: int
    content_type: str
    body: str
    fetched_at: datetime
    etag: str | None = None
    last_modified: str | None = None
    from_cache: bool = False


@dataclass(frozen=True, slots=True)
class NoticeDraft:
    source_id: str
    canonical_url: str
    title: str
    normalized_text: str
    published_at: datetime | None = None
    deadline_at: datetime | None = None
    deadline_confidence: DeadlineConfidence = DeadlineConfidence.UNKNOWN
    location: str = ""
    department: str = ""
    column_name: str = ""
    education_levels: tuple[str, ...] = ()
    grades: tuple[str, ...] = ()
    colleges: tuple[str, ...] = ()
    majors: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    topic_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NoticeRecord:
    id: str
    source_id: str
    canonical_url: str
    title: str
    summary: str
    published_at: datetime | None
    deadline_at: datetime | None
    deadline_confidence: DeadlineConfidence
    location: str = ""
    department: str = ""
    column_name: str = ""
    education_levels: tuple[str, ...] = ()
    grades: tuple[str, ...] = ()
    colleges: tuple[str, ...] = ()
    majors: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    topic_tags: tuple[str, ...] = ()
    content_fingerprint: str = ""
    summary_simhash: int = 0
    parse_state: ParseState = ParseState.PARSED
    fetched_at: datetime | None = None
    proof_status: ProofStatus = ProofStatus.UNVERIFIED_LEGACY
    verified_at: datetime | None = None
    verification_ref: str = ""


@dataclass(frozen=True, slots=True)
class Persona:
    id: str
    label: str
    education_level: str = "不限"
    grade: str = "不限"
    college: str = ""
    major: str = "不限专业"
    campus: str = "不限"
    goals: tuple[str, ...] = ()
    interest_tags: tuple[str, ...] = ()
    custom_keywords: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchHistoryEntry:
    """One recorded "为你推荐" query, used by the clustering learner."""

    id: int
    created_at: datetime
    tags: tuple[str, ...]
    keywords: tuple[str, ...]
    terms: tuple[str, ...]
    persona_id: str | None
    result_count: int
    top_categories: tuple[str, ...] = ()
    top_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ArticleReadEntry:
    notice_id: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    relevance: float
    audience: float
    urgency: float
    authority: float
    freshness: float
    total: float
    # ``audience`` keeps the raw 0..1 value (0.5 when the notice says nothing
    # about who it is for).  ``audience_known`` records whether the persona
    # dimension actually had evidence to work with; when it is False the
    # ranking drops that dimension and normalises the remaining weights.
    audience_known: bool = True
    audience_structural: float = 0.0
    audience_theme: float = 0.0
    audience_structural_known: bool = False
    audience_theme_known: bool = False


@dataclass(frozen=True, slots=True)
class Recommendation:
    notice: NoticeRecord
    score: ScoreBreakdown
    reasons: tuple[str, ...]
    strong_reminder_eligible: bool = False
    match_tier: MatchTier = MatchTier.OTHER
