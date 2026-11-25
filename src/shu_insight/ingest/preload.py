"""Portable, privacy-filtered official evidence. This module performs no HTTP IO."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path

from bs4 import BeautifulSoup

from shu_insight.compliance.detector import redact_public_title, scan_sensitive_text
from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.compliance.policy import evaluate_discovery
from shu_insight.db.repositories import NoticeRepository
from shu_insight.domain.models import (
    DeadlineConfidence,
    FetchedPage,
    NoticeDraft,
    ParseState,
    ProofStatus,
    Verification,
)
from shu_insight.ingest.snapshot import REQUIRED_CHECKS, _aware_datetime, validate_snapshot_item
from shu_insight.mining.fingerprint import simhash
from shu_insight.processing.extract import COLLEGE_TERMS, GRADE_TERMS, MAJOR_TERMS
from shu_insight.processing.public_preview import (
    redact_public_landlines,
    safe_public_preview,
    safe_stored_summary,
)
from shu_insight.sources.base import AttachmentOnly, SourceAdapter, canonicalize_url

_METADATA = {
    "education_levels": frozenset({"本科生", "研究生", "硕士生", "博士生"}),
    "grades": frozenset(GRADE_TERMS),
    "colleges": frozenset(COLLEGE_TERMS),
    "majors": frozenset(MAJOR_TERMS),
    "categories": frozenset({
        "教学教务", "科研学术", "竞赛创新", "奖助评优",
        "就业实习", "升学推免", "校园事务", "讲座活动",
    }),
    "topic_tags": frozenset({"竞赛", "科研", "教务", "就业", "升学", "活动"}),
}
_FIELDS = frozenset({
    "id", "source_id", "canonical_url", "title", "summary", "published_at", "deadline_at",
    "deadline_confidence", "parse_state", "fetched_at", "proof_status", "verified_at",
    "verification_ref", "provenance",
}) | _METADATA.keys()
_ATTACHMENT_URL = "https://bksy.shu.edu.cn/info/1015/322264.htm"
_ATTACHMENT_SHA256 = "1566304fe3b57761c02fed7d574212cf9ae73f3cd5deccfeedfcb5d082e0e79b"
_ATTACHMENT_CHECKS = frozenset({"url", "attachment_only", "sensitivity"})


@dataclass(frozen=True, slots=True)
class PreloadReport:
    imported_count: int = 0
    updated_count: int = 0
    unchanged_count: int = 0
    rejected_count: int = 0
    withdrawn_count: int = 0


def _preview(text: str) -> str:
    node = BeautifulSoup("<div><p>" + escape(safe_stored_summary(text)) + "</p></div>",
                         "lxml").div
    return safe_public_preview(node).strip()


def _policy(registry: Mapping[str, SourceAdapter]) -> OfficialNoticePolicy:
    return OfficialNoticePolicy({key: adapter.config for key, adapter in registry.items()})


def _timestamp(value: object, field: str, *, nullable: bool = False) -> datetime | None:
    if value is None and nullable:
        return None
    parsed = _aware_datetime(value, field)
    if field != "deadline_at" and parsed > datetime.now(UTC):
        raise ValueError(f"{field} is in the future")
    return parsed


def _validate(raw: object, policy: OfficialNoticePolicy) -> dict:
    if not isinstance(raw, dict) or set(raw) != _FIELDS:
        raise ValueError("unexpected preload fields")
    entry = dict(raw)
    for field in ("id", "source_id", "canonical_url", "title", "summary", "verification_ref"):
        if not isinstance(entry[field], str) or entry[field] != entry[field].strip():
            raise ValueError(f"invalid {field}")
    for field, allowed in _METADATA.items():
        values = entry[field]
        if (not isinstance(values, list) or len(values) > 32
                or any(not isinstance(value, str) or len(value) > 80 or value not in allowed
                       for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"invalid public metadata: {field}")
    url = entry["canonical_url"]
    if (not policy.allows(entry["source_id"], url, entry["title"], tuple(entry["topic_tags"]))
            or canonicalize_url(url) != url
            or entry["id"] != str(uuid.uuid5(uuid.NAMESPACE_URL, url))):
        raise ValueError("invalid official identity")
    if (redact_public_title(entry["title"]) != entry["title"]
            or redact_public_landlines(entry["title"]) != entry["title"]
            or scan_sensitive_text(entry["summary"]).blocked
            or _preview(entry["summary"]) != entry["summary"]):
        raise ValueError("unsafe title or preview")
    for field in ("published_at", "deadline_at", "fetched_at", "verified_at"):
        _timestamp(entry[field], field, nullable=field != "verified_at")
    DeadlineConfidence(entry["deadline_confidence"])
    provenance = entry["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("missing acquisition provenance")
    kind = provenance.get("kind")
    if kind == "acquired_database":
        if set(provenance) != {"kind", "capture_time_basis"}:
            raise ValueError("unexpected database provenance fields")
        if provenance["capture_time_basis"] != "original_database_fetched_at":
            raise ValueError("invalid capture time basis")
        if entry["fetched_at"] is None:
            raise ValueError("database acquisition requires original capture time")
    elif kind == "offline_article_review":
        if (set(provenance) != {"kind", "capture_time_basis", "sha256", "checks", "reviewer"}
                or provenance["capture_time_basis"] != "unknown_saved_file_capture_time"
                or not isinstance(provenance["sha256"], str)
                or not re.fullmatch(r"[a-f0-9]{64}", provenance["sha256"])
                or not isinstance(provenance["checks"], dict)
                or set(provenance["checks"]) != REQUIRED_CHECKS
                or not all(value is True for value in provenance["checks"].values())
                or provenance["reviewer"] != "codex_reviewed"
                or entry["fetched_at"] is not None
                or entry["proof_status"] != "verified_snapshot"
                or entry["verification_ref"] != "snapshot:" + provenance["sha256"]):
            raise ValueError("invalid offline review provenance")
    else:
        raise ValueError("untrusted provenance")
    proof, state, reference = entry["proof_status"], entry["parse_state"], entry["verification_ref"]
    if proof == "verified_listing" and state == "listing_only":
        if (not re.fullmatch(r"listing:[a-f0-9]{64}", reference)
                or entry["summary"] or entry["deadline_at"] is not None
                or entry["deadline_confidence"] != "unknown"
                or any(entry[field] for field in _METADATA)
                or kind != "acquired_database"):
            raise ValueError("listing cannot assert body or deadline evidence")
        source = policy.sources[entry["source_id"]]
        if not evaluate_discovery(source, url, entry["title"]).allowed:
            raise ValueError("listing title or URL is outside public discovery policy")
    elif state == "parsed" and proof in {"verified_live", "verified_snapshot"}:
        pattern = r"live:[a-f0-9]{64}" if proof == "verified_live" else r"snapshot:[a-z0-9-]+"
        if not re.fullmatch(pattern, reference):
            raise ValueError("invalid parsed acquisition proof")
    else:
        raise ValueError("preload requires acquired official evidence")
    return entry


def _import_listing(entry: dict, repository: NoticeRepository, verified: datetime) -> str:
    """Keep the acquired list digest; the source DB does not retain its URL.

    upsert_listing requires the observed list URL. Reconstructing one from a
    digest would fabricate provenance, so reuse its narrow persistence contract
    with the already validated original digest instead.
    """
    reference = entry["verification_ref"]
    fingerprint = hashlib.sha256(json.dumps({
        "source_id": entry["source_id"], "canonical_url": entry["canonical_url"],
        "title": entry["title"], "listing_ref": reference,
    }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    payload = {key: entry[key] for key in (
        "title", "published_at", "summary", "deadline_at", "deadline_confidence",
        "proof_status", "verified_at", "verification_ref",
    )}
    with repository.database.transaction() as connection:
        repository._ensure_source(connection, entry["source_id"], verified)
        existing = connection.execute("SELECT id FROM notices WHERE canonical_url=?",
                                      (entry["canonical_url"],)).fetchone()
        notice_id = existing["id"] if existing else entry["id"]
        connection.execute(
            """INSERT INTO notices(
                id, canonical_url, source_id, title, published_at, deadline_at,
                deadline_confidence, summary, content_fingerprint, summary_simhash,
                parse_state, fetched_at, proof_status, verified_at, verification_ref
            ) VALUES (?, ?, ?, ?, ?, NULL, 'unknown', '', ?, ?, 'listing_only', ?,
                      'verified_listing', ?, ?)
            ON CONFLICT(canonical_url) DO UPDATE SET
                title=excluded.title, published_at=excluded.published_at, deadline_at=NULL,
                deadline_confidence='unknown', summary='', location='', department='',
                column_name='', education_levels_json='[]', grades_json='[]',
                colleges_json='[]', majors_json='[]', categories_json='[]', topic_tags_json='[]',
                content_fingerprint=excluded.content_fingerprint,
                summary_simhash=excluded.summary_simhash, parse_state='listing_only',
                fetched_at=excluded.fetched_at, proof_status='verified_listing',
                verified_at=excluded.verified_at, verification_ref=excluded.verification_ref
            """,
            (notice_id, entry["canonical_url"], entry["source_id"], entry["title"],
             entry["published_at"], fingerprint, simhash(""), entry["fetched_at"],
             entry["verified_at"], reference),
        )
        repository._insert_version(connection, notice_id, fingerprint, payload, verified)
    return notice_id


def export_preload(
    source_database: Path, output: Path, registry: Mapping[str, SourceAdapter], *,
    reviewed_entries: Sequence[dict] = (),
    reviewed_exclusions: Sequence[dict] = (),
) -> PreloadReport:
    """Export every suitable record, never initializing or writing the source DB."""
    source_database, output = Path(source_database).resolve(), Path(output).resolve()
    if source_database == output:
        raise ValueError("manifest must not overwrite source database")
    policy = _policy(registry)
    exclusions = [_validate_exclusion(raw, policy) for raw in reviewed_exclusions]
    excluded_urls = {raw["canonical_url"] for raw in exclusions}
    entries: list[dict] = []
    rejected = 0
    with closing(sqlite3.connect(source_database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.row_factory = sqlite3.Row
        for row in connection.execute("SELECT * FROM notices ORDER BY source_id, canonical_url"):
            try:
                record = NoticeRepository._record_from_row(row)
                if not policy.browseable(record):
                    raise ValueError("not browseable")
                listing = record.parse_state == ParseState.LISTING_ONLY
                if listing and record.canonical_url in excluded_urls:
                    raise ValueError("individually reviewed attachment-only listing")
                entry = {key: row[key] for key in _FIELDS - {"provenance"} - _METADATA.keys()}
                entry.update({key: [] if listing else list(getattr(record, key))
                              for key in _METADATA})
                entry.update(
                    title=redact_public_title(record.title),
                    summary="" if listing else _preview(record.summary),
                    deadline_at=None if listing else row["deadline_at"],
                    deadline_confidence="unknown" if listing else row["deadline_confidence"],
                    provenance={"kind": "acquired_database",
                                "capture_time_basis": "original_database_fetched_at"},
                )
                entries.append(_validate(entry, policy))
            except (ValueError, TypeError, KeyError):
                rejected += 1
    entries.extend(_validate(entry, policy) for entry in reviewed_entries)
    if len({entry["canonical_url"] for entry in entries}) != len(entries):
        raise ValueError("duplicate preload identity")
    document = {"schema_version": 1, "kind": "official_offline_preload",
                "exported_at": datetime.now(UTC).isoformat(), "entries": entries,
                "attachment_exclusions": exclusions}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    return PreloadReport(imported_count=len(entries), rejected_count=rejected)


def import_preload(manifest: Path, repository: NoticeRepository) -> PreloadReport:
    """Merge reviewed evidence without HTTP, downgrades, or tombstone resurrection."""
    if repository.policy is None:
        raise RuntimeError("official notice policy is required")
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if (not isinstance(payload, dict) or payload.get("schema_version") != 1
            or payload.get("kind") != "official_offline_preload"
            or not isinstance(payload.get("entries"), list)):
        raise ValueError("invalid official preload manifest")
    imported = updated = unchanged = rejected = 0
    withdrawn = 0
    raw_exclusions = payload.get("attachment_exclusions", [])
    if not isinstance(raw_exclusions, list):
        raise ValueError("attachment exclusions must be a list")
    for raw in raw_exclusions:
        try:
            exclusion = _validate_exclusion(raw, repository.policy)
        except (ValueError, TypeError, KeyError):
            rejected += 1
            continue
        before = repository.find_by_url(exclusion["canonical_url"])
        if (before is not None and before.source_id == exclusion["source_id"]
                and before.parse_state == ParseState.LISTING_ONLY
                and before.proof_status == ProofStatus.VERIFIED_LISTING):
            withdrawn += int(repository.withdraw_listing(exclusion["canonical_url"]))
    seen: set[str] = set()
    for raw in payload["entries"]:
        try:
            entry = _validate(raw, repository.policy)
            url = entry["canonical_url"]
            if url in seen:
                raise ValueError("duplicate preload identity")
            seen.add(url)
        except (ValueError, TypeError, KeyError):
            rejected += 1
            continue
        before = repository.find_by_url(url)
        listing = entry["parse_state"] == "listing_only"
        verified = _aware_datetime(entry["verified_at"], "verified_at")
        if before is not None and (
            before.proof_status == ProofStatus.ATTACHMENT_EXCLUDED
            or (listing and before.parse_state == ParseState.PARSED
                and before.proof_status in {
                    ProofStatus.VERIFIED_LIVE, ProofStatus.VERIFIED_SNAPSHOT,
                })
            or (before.proof_status == ProofStatus.VERIFIED_LIVE
                and entry["proof_status"] == "verified_snapshot")
            or (before.verified_at is not None and before.verified_at >= verified
                and (listing or before.parse_state == ParseState.PARSED))
        ):
            unchanged += 1
            continue
        if listing:
            notice_id = _import_listing(entry, repository, verified)
        else:
            result = repository.upsert(
                NoticeDraft(entry["source_id"], url, entry["title"], entry["summary"],
                            published_at=_timestamp(entry["published_at"], "published_at",
                                                    nullable=True),
                            deadline_at=_timestamp(entry["deadline_at"], "deadline_at",
                                                   nullable=True),
                            deadline_confidence=DeadlineConfidence(entry["deadline_confidence"]),
                            **{key: tuple(entry[key]) for key in _METADATA}),
                verification=Verification(ProofStatus(entry["proof_status"]), verified,
                                          entry["verification_ref"]),
                summary_override=entry["summary"], refresh_mining=False,
            )
            notice_id = result.notice_id
        # Offline review time is the observation time; unknown saved-file capture
        # is represented in the manifest, never guessed from the file's mtime.
        fetched_at = entry["fetched_at"] or entry["verified_at"]
        with repository.database.transaction() as connection:
            connection.execute("UPDATE notices SET fetched_at=? WHERE id=?",
                               (fetched_at, notice_id))
        after = repository.find_by_url(url)
        imported += int(before is None)
        updated += int(before is not None and before != after)
        unchanged += int(before is not None and before == after)
    if imported or updated:
        repository.refresh_mining()
    return PreloadReport(imported, updated + withdrawn, unchanged, rejected, withdrawn)


def _validate_exclusion(raw: object, policy: OfficialNoticePolicy) -> dict:
    """Only the individually reviewed attachment identity can withdraw a listing."""
    if (not isinstance(raw, dict) or set(raw) != {
        "source_id", "canonical_url", "sha256", "reviewed_at", "reviewer", "checks",
    } or raw["source_id"] != "undergraduate" or raw["canonical_url"] != _ATTACHMENT_URL
            or raw["sha256"] != _ATTACHMENT_SHA256 or raw["reviewer"] != "codex_reviewed"
            or not isinstance(raw["checks"], dict) or set(raw["checks"]) != _ATTACHMENT_CHECKS
            or not all(value is True for value in raw["checks"].values())
            or not policy.allows("undergraduate", _ATTACHMENT_URL, "项目申报通知", ())):
        raise ValueError("unproven attachment exclusion")
    _timestamp(raw["reviewed_at"], "reviewed_at")
    return dict(raw)


def reviewed_attachment_exclusion(path: Path, registry: Mapping[str, SourceAdapter]) -> dict:
    """Confirm the exact saved HTML hash and local AttachmentOnly parser result."""
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != _ATTACHMENT_SHA256:
        raise ValueError("saved attachment page differs from reviewed evidence")
    html = content.decode("utf-8")
    if _ATTACHMENT_URL not in html[:1000]:
        raise ValueError("saved-from URL differs from reviewed attachment page")
    reviewed_at = datetime.now(UTC)
    try:
        registry["undergraduate"].parse_detail(FetchedPage(
            "undergraduate", _ATTACHMENT_URL, _ATTACHMENT_URL, 200, "text/html", html, reviewed_at,
        ))
    except AttachmentOnly:
        return _validate_exclusion({
            "source_id": "undergraduate", "canonical_url": _ATTACHMENT_URL,
            "sha256": _ATTACHMENT_SHA256, "reviewed_at": reviewed_at.isoformat(),
            "reviewer": "codex_reviewed", "checks": dict.fromkeys(_ATTACHMENT_CHECKS, True),
        }, _policy(registry))
    raise ValueError("reviewed attachment page now has parsed body evidence")


def reviewed_computer_article(path: Path, registry: Mapping[str, SourceAdapter]) -> dict:
    """Build the individually reviewed saved article; capture time remains unknown."""
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    expected = "8ce4b15bcd70292dfc7e642f40e90475f7516fe2e2d0e2d26f6a90f94befe549"
    if digest != expected:
        raise ValueError("saved article differs from individually reviewed evidence")
    html = content.decode("utf-8")
    soup = BeautifulSoup(html, "lxml")
    header = soup.select_one(".art-main .art-tit > p")
    if header is None or "2026年09月04日" not in header.get_text():
        raise ValueError("reviewed publication header is absent")
    url = "https://cs.shu.edu.cn/info/1381/51981.htm"
    observed = datetime.now(UTC)
    draft = registry["computer"].parse_detail(FetchedPage(
        "computer", url, url, 200, "text/html", html, observed,
    ))
    # Individually reviewed DOM excerpts exclude the working-group roster and
    # later result/list topics that remain in the adapter's broad preview.
    summary = _reviewed_computer_preview(soup)
    reviewed = {
        "id": "computer-51981", "source_id": "computer", "canonical_url": url,
        "title": draft.title, "published_at": "2026-09-04", "deadline_at": None,
        "summary": summary, "verified_at": observed.isoformat(),
        "reviewer": "codex_reviewed", "checks": dict.fromkeys(REQUIRED_CHECKS, True),
    }
    validated = validate_snapshot_item(reviewed, _policy(registry))
    entry = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, url)), "source_id": "computer",
        "canonical_url": url, "title": validated.title, "summary": validated.summary,
        "published_at": validated.published_at.isoformat(), "deadline_at": None,
        "deadline_confidence": "unknown", "parse_state": "parsed", "fetched_at": None,
        "proof_status": "verified_snapshot", "verified_at": observed.isoformat(),
        "verification_ref": "snapshot:" + digest,
        "provenance": {"kind": "offline_article_review", "sha256": digest,
                       "capture_time_basis": "unknown_saved_file_capture_time",
                       "checks": reviewed["checks"], "reviewer": "codex_reviewed"},
        **{key: list(getattr(draft, key)) for key in _METADATA},
    }
    return _validate(entry, _policy(registry))


def _reviewed_computer_preview(soup: BeautifulSoup) -> str:
    intro = soup.select_one(".v_news_content p.vsbcontent_start")
    rows = [row for row in soup.select(".v_news_content tr")
            if row.find("td") is not None
            and row.find("td").get_text("", strip=True) == "9月4-6日"]
    if intro is None or len(rows) != 1:
        raise ValueError("individually reviewed public excerpts are absent")
    summary = _preview(intro.get_text(" ", strip=True) + " "
                       + rows[0].get_text(" ", strip=True))
    if len(summary) > 400 or scan_sensitive_text(summary).blocked:
        raise ValueError("reviewed public excerpts are unsafe")
    return summary
