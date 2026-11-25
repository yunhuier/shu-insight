"""Import only individually reviewed, official article summaries."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from pathlib import Path

from shu_insight.compliance.detector import scan_sensitive_text
from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository
from shu_insight.domain.models import (
    DeadlineConfidence,
    NoticeDraft,
    ProofStatus,
    Verification,
)

REQUIRED_CHECKS = frozenset({
    "url", "title", "date", "audience", "action", "summary", "sensitivity"
})
SHANGHAI_TZ = timezone(timedelta(hours=8))
_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


@dataclass(frozen=True, slots=True)
class SnapshotItem:
    id: str
    source_id: str
    url: str
    title: str
    published_at: datetime | None
    deadline_at: datetime | None
    deadline_confidence: DeadlineConfidence
    verified_at: datetime
    summary: str


@dataclass(frozen=True, slots=True)
class SnapshotImportReport:
    imported_count: int
    unchanged_count: int
    rejected_count: int


def _aware_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def validate_snapshot_item(
    raw: Mapping[str, object], policy: OfficialNoticePolicy
) -> SnapshotItem:
    checks = raw.get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != REQUIRED_CHECKS
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("snapshot entry lacks completed review checks")
    item_id = raw.get("id")
    source_id = raw.get("source_id")
    url = raw.get("canonical_url")
    title = raw.get("title")
    summary = raw.get("summary")
    if not isinstance(item_id, str) or not _ID.fullmatch(item_id):
        raise ValueError("snapshot id must be a stable slug")
    if not all(isinstance(value, str) and value == value.strip() and value
               for value in (source_id, url, title)):
        raise ValueError("snapshot source, URL, and title must be nonempty strings")
    if not policy.allows(source_id, url, title, ()):
        raise ValueError("snapshot URL/title outside reviewed official source")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("snapshot summary is empty")
    summary = summary.strip()
    if scan_sensitive_text(title + "\n" + summary).blocked:
        raise ValueError("snapshot title or summary contains sensitive content")
    if raw.get("reviewer") != "codex_reviewed":
        raise ValueError("reviewer must state AI-assisted review provenance")
    verified_at = _aware_datetime(raw.get("verified_at"), "verified_at")
    if verified_at > datetime.now(UTC):
        raise ValueError("snapshot verified_at must be an observed time")
    published = raw.get("published_at")
    if published is None:
        published_at = None
    else:
        if not isinstance(published, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", published):
            raise ValueError("published_at must be an ISO date or null")
        published_at = datetime.combine(date.fromisoformat(published), time.min, SHANGHAI_TZ)
    deadline = raw.get("deadline_at")
    if deadline is None:
        deadline_at = None
        deadline_confidence = DeadlineConfidence.UNKNOWN
    elif isinstance(deadline, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", deadline):
        deadline_at = datetime.combine(date.fromisoformat(deadline), time.min, SHANGHAI_TZ)
        deadline_confidence = DeadlineConfidence.LOW
    else:
        deadline_at = _aware_datetime(deadline, "deadline_at")
        deadline_confidence = DeadlineConfidence.HIGH
    return SnapshotItem(item_id, source_id, url, title, published_at,
                        deadline_at, deadline_confidence, verified_at, summary)


def import_snapshot(
    manifest_path: Path, repository: NoticeRepository
) -> SnapshotImportReport:
    if repository.policy is None:
        raise RuntimeError("official notice policy is required")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("entries"), list)
    ):
        raise ValueError("invalid official snapshot manifest")
    raw_entries = payload["entries"]
    if not all(isinstance(raw, dict) for raw in raw_entries):
        raise ValueError("snapshot entries must be objects")
    items = tuple(validate_snapshot_item(raw, repository.policy) for raw in raw_entries)
    if (
        len({item.id for item in items}) != len(items)
        or len({item.url for item in items}) != len(items)
    ):
        raise ValueError("duplicate snapshot id or URL")
    imported = unchanged = 0
    for item in items:
        draft = NoticeDraft(source_id=item.source_id, canonical_url=item.url,
                            title=item.title, normalized_text=item.summary,
                            published_at=item.published_at, deadline_at=item.deadline_at,
                            deadline_confidence=item.deadline_confidence)
        proof = Verification(ProofStatus.VERIFIED_SNAPSHOT, item.verified_at,
                             f"snapshot:{item.id}")
        result = repository.upsert(draft, verification=proof,
                                   summary_override=item.summary)
        imported += int(result.created or result.updated)
        unchanged += int(not result.created and not result.updated)
    return SnapshotImportReport(imported, unchanged, 0)
