from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from shu_insight.compliance.policy import BLOCKED_SUFFIXES, evaluate_discovery, evaluate_request
from shu_insight.domain.models import NoticeRecord, ParseState, ProofStatus, SourceConfig

_ENCODED_PATH_ESCAPE = re.compile(r"%(?:2e|2f|5c)", re.IGNORECASE)
_VERIFIED = {ProofStatus.VERIFIED_SNAPSHOT, ProofStatus.VERIFIED_LIVE}


class OfficialNoticePolicy:
    """The admission and read boundary for verified official article URLs."""

    def __init__(self, sources: Mapping[str, SourceConfig]) -> None:
        self.sources = dict(sources)

    def allows(self, source_id: str, url: str, title: str, tags: tuple[str, ...]) -> bool:
        source = self.sources.get(source_id)
        if source is None or not title.strip() or "[演示]" in title:
            return False
        if "synthetic_demo" in tags or not url or url != url.strip():
            return False
        if not evaluate_request(source, url).allowed:
            return False
        if any(ord(char) < 33 for char in url):
            return False
        try:
            parsed = urlsplit(url)
            configured = urlsplit(source.base_url)
            port = parsed.port
        except ValueError:
            return False
        if (
            parsed.scheme.lower() != "https"
            or parsed.hostname is None
            or parsed.hostname.lower() != (configured.hostname or "").lower()
            or parsed.username is not None
            or parsed.password is not None
            or port is not None
            or "\\" in parsed.path
            or _ENCODED_PATH_ESCAPE.search(parsed.path)
        ):
            return False
        segments = parsed.path.split("/")
        if "." in segments or ".." in segments:
            return False
        if not parsed.path.startswith("/info/") or parsed.path == "/info/":
            return False
        return PurePosixPath(parsed.path).suffix.lower() not in BLOCKED_SUFFIXES

    def visible(self, record: NoticeRecord) -> bool:
        return (
            record.proof_status in _VERIFIED
            and record.parse_state == ParseState.PARSED
            and record.verified_at is not None
            and record.verified_at.tzinfo is not None
            and record.verified_at.utcoffset() is not None
            and bool(record.verification_ref.strip())
            and self.allows(record.source_id, record.canonical_url, record.title, record.topic_tags)
        )

    def allows_list_page(self, source_id: str, list_url: str) -> bool:
        source = self.sources.get(source_id)
        if source is None or not evaluate_request(source, list_url).allowed:
            return False
        parsed = urlsplit(list_url)
        if parsed.query or parsed.fragment:
            return False
        if list_url == source.list_url:
            return True
        return (
            source.pagination_reviewed_on is not None
            and bool(source.pagination_url_pattern)
            and re.fullmatch(source.pagination_url_pattern, parsed.path) is not None
        )

    def browseable(self, record: NoticeRecord) -> bool:
        if self.visible(record):
            return True
        source = self.sources.get(record.source_id)
        return (
            source is not None
            and record.proof_status == ProofStatus.VERIFIED_LISTING
            and record.parse_state == ParseState.LISTING_ONLY
            and record.verified_at is not None
            and record.verified_at.tzinfo is not None
            and record.verified_at.utcoffset() is not None
            and re.fullmatch(r"listing:[0-9a-f]{64}", record.verification_ref) is not None
            and self.allows(record.source_id, record.canonical_url, record.title, record.topic_tags)
            and evaluate_discovery(source, record.canonical_url, record.title).allowed
        )
