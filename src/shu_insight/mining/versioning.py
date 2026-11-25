from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from shu_insight.domain.models import NoticeRecord

_VERSIONED_FIELDS = (
    "title",
    "summary",
    "published_at",
    "deadline_at",
    "deadline_confidence",
    "location",
    "department",
    "column_name",
    "education_levels",
    "grades",
    "colleges",
    "majors",
    "categories",
    "topic_tags",
)


@dataclass(frozen=True, slots=True)
class FieldChange:
    field: str
    before: Any
    after: Any


@dataclass(frozen=True, slots=True)
class VersionDiff:
    previous_fingerprint: str
    current_fingerprint: str
    changes: tuple[FieldChange, ...]

    @property
    def changed_fields(self) -> frozenset[str]:
        return frozenset(change.field for change in self.changes)

    @property
    def has_changes(self) -> bool:
        return bool(self.changes)


def _derived_fingerprint(notice: NoticeRecord) -> str:
    values = {field: getattr(notice, field) for field in _VERSIONED_FIELDS}
    payload = json.dumps(values, default=str, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def compare_versions(previous: NoticeRecord, current: NoticeRecord) -> VersionDiff:
    if previous.canonical_url != current.canonical_url:
        raise ValueError("versions must share the same canonical URL")
    changes = tuple(
        FieldChange(field, getattr(previous, field), getattr(current, field))
        for field in _VERSIONED_FIELDS
        if getattr(previous, field) != getattr(current, field)
    )
    return VersionDiff(
        previous_fingerprint=previous.content_fingerprint or _derived_fingerprint(previous),
        current_fingerprint=current.content_fingerprint or _derived_fingerprint(current),
        changes=changes,
    )
