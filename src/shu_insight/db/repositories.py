from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.compliance.policy import evaluate_discovery, evaluate_request
from shu_insight.db.schema import Database
from shu_insight.domain.models import (
    ArticleReadEntry,
    DeadlineConfidence,
    DiscoveredNotice,
    NoticeDraft,
    NoticeRecord,
    ParseState,
    Persona,
    ProofStatus,
    SearchHistoryEntry,
    SourceConfig,
    Verification,
)
from shu_insight.mining.deduplicate import assign_clusters
from shu_insight.mining.fingerprint import simhash
from shu_insight.mining.versioning import compare_versions
from shu_insight.processing.dates import parse_publication, utc_datetime


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    return utc_datetime(value).isoformat() if value is not None else None


def _parse_datetime(value: str | None) -> datetime | None:
    return utc_datetime(datetime.fromisoformat(value)) if value else None


def _parse_verified_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return utc_datetime(parsed)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _summary(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1]}…"


@dataclass(frozen=True, slots=True)
class UpsertResult:
    notice_id: str
    created: bool
    updated: bool


class NoticeRepository:
    def __init__(self, database: Database, policy: OfficialNoticePolicy | None = None) -> None:
        self.database = database
        self.policy = policy

    def upsert_listing(
        self, item: DiscoveredNotice, *, list_url: str, checked_at: datetime,
    ) -> UpsertResult:
        """Persist public list evidence without claiming a fetched or parsed article body."""
        if self.policy is None:
            raise RuntimeError("official notice policy is required for listing writes")
        if checked_at.tzinfo is None or checked_at.utcoffset() is None:
            raise ValueError("listing evidence time must be timezone aware")
        if (
            not self.policy.allows_list_page(item.source_id, list_url)
            or not self.policy.allows(item.source_id, item.url, item.title, ())
            or not evaluate_discovery(
                self.policy.sources[item.source_id], item.url, item.title,
            ).allowed
        ):
            raise ValueError("listing is outside the official list/article policy")
        notice_id = str(uuid.uuid5(uuid.NAMESPACE_URL, item.url))
        checked_iso = _iso(checked_at)
        reference = "listing:" + hashlib.sha256(list_url.encode("utf-8")).hexdigest()
        # Identity excludes time/date hints so a repeated list cannot revive an
        # attachment tombstone merely because a clock or date hint changed.
        fingerprint = hashlib.sha256(_json({
            "source_id": item.source_id, "canonical_url": item.url,
            "title": item.title, "listing_ref": reference,
        }).encode("utf-8")).hexdigest()
        payload = {
            "title": item.title, "published_at": _iso(parse_publication(item.published_hint)),
            "summary": "", "deadline_at": None,
            "deadline_confidence": DeadlineConfidence.UNKNOWN.value,
            "proof_status": ProofStatus.VERIFIED_LISTING.value,
            "verified_at": checked_iso, "verification_ref": reference,
        }
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM notices WHERE canonical_url = ?", (item.url,),
            ).fetchone()
            if existing is not None:
                notice_id = existing["id"]
                if existing["proof_status"] in (
                    ProofStatus.VERIFIED_LIVE.value, ProofStatus.VERIFIED_SNAPSHOT.value,
                ) or (
                    existing["proof_status"] == ProofStatus.ATTACHMENT_EXCLUDED.value
                    and existing["content_fingerprint"] == fingerprint
                ):
                    return UpsertResult(notice_id, created=False, updated=False)
            self._ensure_source(connection, item.source_id, checked_at)
            changed = (
                existing is None or existing["content_fingerprint"] != fingerprint
                or existing["published_at"] != payload["published_at"]
                or existing["proof_status"] != ProofStatus.VERIFIED_LISTING.value
            )
            connection.execute(
                """
                INSERT INTO notices(
                  id, canonical_url, source_id, title, published_at, deadline_at,
                  deadline_confidence, summary, content_fingerprint, summary_simhash,
                  parse_state, fetched_at, proof_status, verified_at, verification_ref
                ) VALUES (?, ?, ?, ?, ?, NULL, ?, '', ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_url) DO UPDATE SET
                  title = excluded.title, published_at = excluded.published_at,
                  deadline_at = NULL, deadline_confidence = excluded.deadline_confidence,
                  summary = '', location = '', department = '', column_name = '',
                  education_levels_json = '[]', grades_json = '[]', colleges_json = '[]',
                  majors_json = '[]', categories_json = '[]', topic_tags_json = '[]',
                  content_fingerprint = excluded.content_fingerprint,
                  summary_simhash = excluded.summary_simhash, parse_state = excluded.parse_state,
                  fetched_at = excluded.fetched_at, proof_status = excluded.proof_status,
                  verified_at = excluded.verified_at, verification_ref = excluded.verification_ref
                """,
                (notice_id, item.url, item.source_id, item.title, payload["published_at"],
                 DeadlineConfidence.UNKNOWN.value, fingerprint, simhash(""),
                 ParseState.LISTING_ONLY.value, checked_iso,
                 ProofStatus.VERIFIED_LISTING.value, checked_iso, reference),
            )
            if changed:
                self._insert_version(connection, notice_id, fingerprint, payload, checked_at)
            return UpsertResult(notice_id, created=existing is None,
                                updated=existing is not None and changed)

    def withdraw_listing(self, canonical_url: str) -> bool:
        """Keep a list-only attachment tombstone while removing its visible proof."""
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE notices SET proof_status = ?, verified_at = NULL, verification_ref = '' "
                "WHERE canonical_url = ? AND parse_state = ? AND proof_status = ?",
                (ProofStatus.ATTACHMENT_EXCLUDED.value, canonical_url,
                 ParseState.LISTING_ONLY.value, ProofStatus.VERIFIED_LISTING.value),
            )
            return cursor.rowcount > 0

    def refresh_live_verification(self, canonical_url: str, verified_at: datetime) -> bool:
        """Record a 304 recheck only for an already verified live article."""
        if verified_at.tzinfo is None or verified_at.utcoffset() is None:
            raise ValueError("verification time must be timezone aware")
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT source_id, title, topic_tags_json, proof_status, verification_ref "
                "FROM notices WHERE canonical_url = ?", (canonical_url,)
            ).fetchone()
            if (row is None or row["proof_status"] != ProofStatus.VERIFIED_LIVE.value
                    or not row["verification_ref"].startswith("live:")
                    or self.policy is None
                    or not self.policy.allows(row["source_id"], canonical_url, row["title"],
                                              tuple(json.loads(row["topic_tags_json"])))):
                return False
            connection.execute(
                "UPDATE notices SET verified_at = ? WHERE canonical_url = ?",
                (_iso(verified_at), canonical_url),
            )
            return True

    def upsert(
        self,
        draft: NoticeDraft,
        *,
        verification: Verification | None = None,
        summary_override: str | None = None,
        refresh_mining: bool = True,
    ) -> UpsertResult:
        if verification is not None:
            if (
                verification.status not in (
                    ProofStatus.VERIFIED_SNAPSHOT, ProofStatus.VERIFIED_LIVE
                )
                or verification.verified_at.tzinfo is None
                or verification.verified_at.utcoffset() is None
                or not verification.reference.strip()
            ):
                raise ValueError("a verified notice requires valid proof and aware time")
            if self.policy is None:
                raise RuntimeError("official notice policy is required for verified writes")
            if not self.policy.allows(
                draft.source_id, draft.canonical_url, draft.title, draft.topic_tags
            ):
                raise ValueError("notice is outside the official article policy")
        notice_id = str(uuid.uuid5(uuid.NAMESPACE_URL, draft.canonical_url))
        now = _utc_now()
        summary = (
            _summary(draft.normalized_text)
            if summary_override is None
            else summary_override.strip()
        )
        proof_status = verification.status if verification else ProofStatus.UNVERIFIED_LEGACY
        verified_at = _iso(verification.verified_at) if verification else None
        verification_ref = verification.reference.strip() if verification else ""
        payload = {
            "title": draft.title,
            "published_at": _iso(draft.published_at),
            "deadline_at": _iso(draft.deadline_at),
            "deadline_confidence": draft.deadline_confidence.value,
            "location": draft.location,
            "department": draft.department,
            "column_name": draft.column_name,
            "education_levels": draft.education_levels,
            "grades": draft.grades,
            "colleges": draft.colleges,
            "majors": draft.majors,
            "categories": draft.categories,
            "topic_tags": draft.topic_tags,
            "summary": summary,
            "proof_status": proof_status.value,
            "verified_at": verified_at,
            "verification_ref": verification_ref,
        }
        fingerprint = hashlib.sha256(
            _json({**{key: value for key, value in payload.items() if key not in (
                "proof_status", "verified_at", "verification_ref"
            )}, "body_hash": hashlib.sha256(
                draft.normalized_text.encode("utf-8")).hexdigest()}).encode("utf-8")
        ).hexdigest()

        with self.database.transaction() as connection:
            self._ensure_source(connection, draft.source_id, now)
            existing = connection.execute(
                "SELECT * FROM notices WHERE canonical_url = ?",
                (draft.canonical_url,),
            ).fetchone()
            if (
                existing is not None
                and verification is not None
                and verification.status == ProofStatus.VERIFIED_SNAPSHOT
                and existing["proof_status"] == ProofStatus.VERIFIED_LIVE
            ):
                return UpsertResult(existing["id"], created=False, updated=False)
            if existing is not None and existing["content_fingerprint"] == fingerprint:
                if verification is not None:
                    connection.execute(
                        "UPDATE notices SET proof_status = ?, verified_at = ?, "
                        "verification_ref = ? WHERE id = ?",
                        (proof_status.value, verified_at, verification_ref, existing["id"]),
                    )
                return UpsertResult(existing["id"], created=False, updated=False)

            if existing is None:
                connection.execute(
                    """
                    INSERT INTO notices(
                      id, canonical_url, source_id, title, published_at, deadline_at,
                      deadline_confidence, summary, location, department, column_name,
                      education_levels_json, grades_json, colleges_json, majors_json,
                      categories_json, topic_tags_json, content_fingerprint,
                      summary_simhash, parse_state, fetched_at,
                      proof_status, verified_at, verification_ref
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        notice_id,
                        draft.canonical_url,
                        draft.source_id,
                        draft.title,
                        _iso(draft.published_at),
                        _iso(draft.deadline_at),
                        draft.deadline_confidence.value,
                        summary,
                        draft.location,
                        draft.department,
                        draft.column_name,
                        _json(draft.education_levels),
                        _json(draft.grades),
                        _json(draft.colleges),
                        _json(draft.majors),
                        _json(draft.categories),
                        _json(draft.topic_tags),
                        fingerprint,
                        simhash(summary),
                        ParseState.PARSED.value,
                        now.isoformat(),
                        proof_status.value,
                        verified_at,
                        verification_ref,
                    ),
                )
                created = True
            else:
                notice_id = existing["id"]
                connection.execute(
                    """
                    UPDATE notices
                    SET title = ?, published_at = ?, deadline_at = ?,
                        deadline_confidence = ?, summary = ?, location = ?, department = ?,
                        column_name = ?, education_levels_json = ?, grades_json = ?,
                        colleges_json = ?, majors_json = ?, categories_json = ?,
                        topic_tags_json = ?, content_fingerprint = ?, summary_simhash = ?,
                        parse_state = ?, fetched_at = ?, proof_status = ?,
                        verified_at = ?, verification_ref = ?
                    WHERE id = ?
                    """,
                    (
                        draft.title,
                        _iso(draft.published_at),
                        _iso(draft.deadline_at),
                        draft.deadline_confidence.value,
                        summary,
                        draft.location,
                        draft.department,
                        draft.column_name,
                        _json(draft.education_levels),
                        _json(draft.grades),
                        _json(draft.colleges),
                        _json(draft.majors),
                        _json(draft.categories),
                        _json(draft.topic_tags),
                        fingerprint,
                        simhash(summary),
                        ParseState.PARSED.value,
                        now.isoformat(),
                        proof_status.value,
                        verified_at,
                        verification_ref,
                        notice_id,
                    ),
                )
                created = False

            payload["changes"] = []
            if existing is not None:
                current = self._record_from_row(connection.execute(
                    "SELECT * FROM notices WHERE id = ?", (notice_id,)
                ).fetchone())
                diff = compare_versions(self._record_from_row(existing), current)
                payload["changes"] = [
                    {"field": change.field,
                     "before": _iso(change.before) if isinstance(change.before, datetime)
                     else change.before,
                     "after": _iso(change.after) if isinstance(change.after, datetime)
                     else change.after}
                    for change in diff.changes
                ]
                payload["previous_fingerprint"] = diff.previous_fingerprint
            self._insert_version(connection, notice_id, fingerprint, payload, now)
            if refresh_mining:
                self._refresh_mining(connection, now)
            return UpsertResult(notice_id, created=created, updated=not created)

    def refresh_mining(self) -> None:
        with self.database.transaction() as connection:
            self._refresh_mining(connection, _utc_now())

    def _refresh_mining(self, connection: sqlite3.Connection, now: datetime) -> None:
        records = [self._record_from_row(row) for row in connection.execute(
            "SELECT * FROM notices WHERE parse_state != ? AND proof_status != ? "
            "AND TRIM(summary) != '' ORDER BY id",
            (ParseState.LISTING_ONLY.value, ProofStatus.ATTACHMENT_EXCLUDED.value),
        )]
        assignments = assign_clusters(records)
        by_id = {notice.id: notice for notice in records}
        connection.execute("DELETE FROM duplicate_cluster_members")
        connection.execute("DELETE FROM duplicate_clusters")
        for cluster in assignments:
            confidence = min((e.score for e in cluster.evidence), default=1.0)
            connection.execute(
                "INSERT OR IGNORE INTO duplicate_clusters VALUES (?, ?, ?)",
                (cluster.id, confidence, now.isoformat()),
            )
            for member in cluster.members:
                connection.execute("INSERT INTO duplicate_cluster_members VALUES (?, ?, ?)",
                                   (cluster.id, member.id, "duplicate" if len(cluster.members) > 1
                                    else "singleton"))
        evidence = [item for cluster in assignments for item in cluster.evidence]
        for item in (*evidence, *assignments.possible_duplicates):
            connection.execute(
                "INSERT OR IGNORE INTO duplicate_relations VALUES (?, ?, ?, ?, ?, ?)",
                (item.left_id, item.right_id, by_id[item.left_id].content_fingerprint,
                 by_id[item.right_id].content_fingerprint, _json(asdict(item)), now.isoformat()),
            )

    def detail_evidence(self, notice_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            version_rows = connection.execute(
                "SELECT * FROM notice_versions WHERE notice_id = ? ORDER BY id", (notice_id,)
            ).fetchall()
            relations = connection.execute("""
                SELECT r.* FROM duplicate_relations r
                JOIN notices l ON l.id = r.left_id AND l.content_fingerprint = r.left_fingerprint
                JOIN notices n ON n.id = r.right_id AND n.content_fingerprint = r.right_fingerprint
                WHERE r.left_id = ? OR r.right_id = ?
            """, (notice_id, notice_id)).fetchall()
            by_id = {row["id"]: self._record_from_row(row)
                     for row in connection.execute("SELECT * FROM notices")}
        duplicates, possible = [], []
        for row in relations:
            evidence = json.loads(row["evidence_json"])
            other_id = row["right_id"] if row["left_id"] == notice_id else row["left_id"]
            item = {"notice": by_id[other_id], "evidence": evidence}
            (duplicates if evidence["disposition"] == "duplicate" else possible).append(item)
        return {"duplicates": duplicates, "possible_duplicates": possible,
                "versions": [{**json.loads(row["payload_json"]), "created_at": row["created_at"],
                              "fingerprint": row["content_fingerprint"]} for row in version_rows]}

    def detail_evidence_verified(self, notice_id: str) -> dict[str, Any]:
        empty = {"duplicates": [], "possible_duplicates": [], "versions": []}
        if self.policy is None:
            raise RuntimeError("official notice policy is required for verified reads")
        record = next((row for row in self.list_verified() if row.id == notice_id), None)
        if record is None:
            return empty
        raw = self.detail_evidence(notice_id)
        for key in ("duplicates", "possible_duplicates"):
            empty[key] = [
                item for item in raw[key] if self.policy.visible(item["notice"])
            ]
        previous_verified = False
        for version in raw["versions"]:
            try:
                verified_at = datetime.fromisoformat(version["verified_at"])
            except (KeyError, TypeError, ValueError):
                previous_verified = False
                continue
            version_verified = (
                version.get("proof_status") in (
                    ProofStatus.VERIFIED_SNAPSHOT, ProofStatus.VERIFIED_LIVE
                )
                and verified_at.tzinfo is not None
                and verified_at.utcoffset() is not None
                and str(version.get("verification_ref", "")).strip()
                and self.policy.allows(
                    record.source_id, record.canonical_url,
                    version.get("title", ""), tuple(version.get("topic_tags", ())),
                )
            )
            if version_verified:
                visible_version = dict(version)
                if not previous_verified:
                    visible_version["changes"] = [
                        {key: value for key, value in change.items() if key != "before"}
                        for change in version.get("changes", [])
                        if isinstance(change, dict)
                    ]
                    visible_version.pop("previous_fingerprint", None)
                empty["versions"].append(visible_version)
            previous_verified = version_verified
        return empty

    def _ensure_source(
        self, connection: sqlite3.Connection, source_id: str, created_at: datetime
    ) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO sources(id, name, base_url, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (source_id, source_id, "https://placeholder.invalid/", created_at.isoformat()),
        )

    def _insert_version(
        self,
        connection: sqlite3.Connection,
        notice_id: str,
        fingerprint: str,
        payload: Mapping[str, Any],
        created_at: datetime,
    ) -> None:
        connection.execute(
            """
            INSERT INTO notice_versions(notice_id, content_fingerprint, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (notice_id, fingerprint, _json(payload), created_at.isoformat()),
        )

    def count(self) -> int:
        with self.database.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM notices").fetchone()[0])

    def list_all(self) -> list[NoticeRecord]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM notices ORDER BY published_at DESC, id"
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def list_verified(self) -> list[NoticeRecord]:
        if self.policy is None:
            raise RuntimeError("official notice policy is required for verified reads")
        return [row for row in self.list_all() if self.policy.visible(row)]

    def list_browseable(self) -> list[NoticeRecord]:
        if self.policy is None:
            raise RuntimeError("official notice policy is required for browseable reads")
        return [row for row in self.list_all() if self.policy.browseable(row)]

    def list_live_for_recheck(
        self, source_id: str, exclude_urls: Collection[str],
    ) -> list[NoticeRecord]:
        if self.policy is None:
            raise RuntimeError("official notice policy is required for verified reads")
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM notices WHERE source_id = ? AND proof_status = ? "
                "ORDER BY verified_at ASC, canonical_url ASC",
                (source_id, ProofStatus.VERIFIED_LIVE),
            ).fetchall()
        records = [self._record_from_row(row) for row in rows]
        return [record for record in records if record.canonical_url not in exclude_urls
                and self.policy.visible(record)]

    def list_parsed_for_recheck(
        self, source_id: str, exclude_urls: Collection[str],
    ) -> list[NoticeRecord]:
        """Both live and snapshot proofs share the manual candidate allowance."""
        return sorted(
            (record for record in self.list_verified()
             if record.source_id == source_id and record.canonical_url not in exclude_urls),
            key=lambda record: (record.verified_at, record.canonical_url),
        )

    def find_by_url(self, canonical_url: str) -> NoticeRecord | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM notices WHERE canonical_url = ?", (canonical_url,)
            ).fetchone()
        return self._record_from_row(row) if row is not None else None

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> NoticeRecord:
        return NoticeRecord(
            id=row["id"],
            source_id=row["source_id"],
            canonical_url=row["canonical_url"],
            title=row["title"],
            summary=row["summary"],
            published_at=_parse_datetime(row["published_at"]),
            deadline_at=_parse_datetime(row["deadline_at"]),
            deadline_confidence=DeadlineConfidence(row["deadline_confidence"]),
            location=row["location"],
            department=row["department"],
            column_name=row["column_name"],
            education_levels=tuple(json.loads(row["education_levels_json"])),
            grades=tuple(json.loads(row["grades_json"])),
            colleges=tuple(json.loads(row["colleges_json"])),
            majors=tuple(json.loads(row["majors_json"])),
            categories=tuple(json.loads(row["categories_json"])),
            topic_tags=tuple(json.loads(row["topic_tags_json"])),
            content_fingerprint=row["content_fingerprint"],
            summary_simhash=row["summary_simhash"],
            parse_state=ParseState(row["parse_state"]),
            fetched_at=_parse_datetime(row["fetched_at"]),
            proof_status=ProofStatus(row["proof_status"]),
            verified_at=_parse_verified_at(row["verified_at"]),
            verification_ref=row["verification_ref"],
        )


class PersonaRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def save(self, persona: Persona) -> None:
        now = _utc_now().isoformat()
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO personas(id, label, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  label = excluded.label,
                  payload_json = excluded.payload_json,
                  updated_at = excluded.updated_at
                """,
                (persona.id, persona.label, _json(asdict(persona)), now),
            )

    def list_all(self) -> list[Persona]:
        with self.database.connect() as connection:
            rows = connection.execute("SELECT payload_json FROM personas ORDER BY label").fetchall()
        personas: list[Persona] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            for field in ("goals", "interest_tags", "custom_keywords"):
                payload[field] = tuple(payload.get(field, ()))
            personas.append(Persona(**payload))
        return personas

    def delete_all(self) -> None:
        with self.database.transaction() as connection:
            connection.execute("DELETE FROM personas")

    def delete(self, persona_id: str) -> bool:
        with self.database.transaction() as connection:
            cursor = connection.execute("DELETE FROM personas WHERE id = ?", (persona_id,))
        return cursor.rowcount > 0


class ArticleReadRepository:
    """One first-click record per notice, shared by title and original-link clicks."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def record(self, notice_id: str, *, now: datetime | None = None) -> bool:
        with self.database.transaction() as connection:
            result = connection.execute(
                "INSERT INTO article_reads(notice_id, first_read_at) "
                "SELECT id, ? FROM notices WHERE id = ? "
                "ON CONFLICT(notice_id) DO NOTHING",
                (_iso(now or _utc_now()), notice_id),
            )
            return result.rowcount > 0

    def entries(self, limit: int = 200) -> list[ArticleReadEntry]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM (SELECT rowid AS read_order, notice_id, first_read_at "
                "FROM article_reads ORDER BY rowid DESC LIMIT ?) ORDER BY read_order",
                (limit,),
            ).fetchall()
        return [ArticleReadEntry(row["notice_id"], datetime.fromisoformat(row["first_read_at"]))
                for row in rows]

    def count(self) -> int:
        with self.database.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM article_reads").fetchone()[0])


class SearchHistoryRepository:
    """Stores the local user's own "为你推荐" queries for the clustering learner."""

    DUPLICATE_WINDOW = timedelta(hours=6)

    def __init__(self, database: Database) -> None:
        self.database = database

    def record(
        self,
        *,
        tags: tuple[str, ...],
        keywords: tuple[str, ...],
        terms: tuple[str, ...],
        persona_id: str | None,
        result_count: int,
        top_categories: tuple[str, ...] = (),
        top_tags: tuple[str, ...] = (),
        now: datetime | None = None,
    ) -> bool:
        """Insert one query; identical reruns inside the window are ignored."""
        moment = now or _utc_now()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        signature = (
            tuple(tags), tuple(keywords), tuple(terms), persona_id,
        )
        with self.database.transaction() as connection:
            latest = connection.execute(
                "SELECT created_at, tags_json, keywords_json, terms_json, persona_id "
                "FROM search_history ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if latest is not None:
                try:
                    previous_at = datetime.fromisoformat(latest["created_at"])
                except ValueError:
                    previous_at = None
                previous = (
                    tuple(json.loads(latest["tags_json"])),
                    tuple(json.loads(latest["keywords_json"])),
                    tuple(json.loads(latest["terms_json"])),
                    latest["persona_id"],
                )
                if (
                    previous == signature
                    and previous_at is not None
                    and abs(moment - previous_at) <= self.DUPLICATE_WINDOW
                ):
                    return False
            connection.execute(
                """
                INSERT INTO search_history(
                  created_at, tags_json, keywords_json, terms_json, persona_id,
                  result_count, top_categories_json, top_tags_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    moment.isoformat(),
                    _json(list(tags)),
                    _json(list(keywords)),
                    _json(list(terms)),
                    persona_id,
                    result_count,
                    _json(list(top_categories)),
                    _json(list(top_tags)),
                ),
            )
        return True

    def entries(self, limit: int = 200) -> list[SearchHistoryEntry]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM (
                    SELECT * FROM search_history ORDER BY id DESC LIMIT ?
                ) ORDER BY id
                """,
                (limit,),
            ).fetchall()
        return [
            SearchHistoryEntry(
                id=row["id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                tags=tuple(json.loads(row["tags_json"])),
                keywords=tuple(json.loads(row["keywords_json"])),
                terms=tuple(json.loads(row["terms_json"])),
                persona_id=row["persona_id"],
                result_count=row["result_count"],
                top_categories=tuple(json.loads(row["top_categories_json"])),
                top_tags=tuple(json.loads(row["top_tags_json"])),
            )
            for row in rows
        ]

    def count(self) -> int:
        with self.database.connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM search_history").fetchone()
        return int(row[0])

    def clear(self) -> None:
        with self.database.transaction() as connection:
            connection.execute("DELETE FROM search_history")


SAFE_SOURCE_STATES = frozenset({
    "success", "not_modified", "rate_limited", "forbidden", "robots_disallowed",
    "robots_unavailable", "login_required", "login_or_captcha",
    "robots_not_found", "robots_forbidden", "robots_rate_limited",
    "robots_other_4xx", "robots_server_error", "robots_redirect",
    "robots_unexpected_status", "robots_wrong_content_type",
    "robots_html_or_login", "robots_missing_directives",
    "authentication_required", "circuit_open", "redirect_not_allowed",
    "not_modified_without_body", "error", "unknown", "not_checked",
    "synthetic_demo", "structure_changed", "path_not_allowed",
    "attachment_blocked", "unsupported_content_type", "non_html", "http_error",
    "partial", "config_review_required", "interrupted", "request_budget_exhausted",
    "connection_failed", "connection_timeout",
    "unresolved_redirect", "invalid_url", "https_required", "host_not_allowed",
})
SAFE_TRANSPORT_CODES = frozenset(
    f"transport_{stage}_{kind}"
    for stage in ("robots", "list", "detail")
    for kind in (
        "proxy", "connect", "connect_timeout", "read", "read_timeout", "other",
        "remote_protocol", "local_protocol", "write", "write_timeout", "pool_timeout",
        "close", "unsupported_protocol",
    )
)
SAFE_SKIP_REASONS = frozenset({
    "sensitive_content", "structure_changed", "content_unchanged", "unchanged_list",
    "unchanged_detail", "path_not_allowed", "attachment_blocked", "other",
    "queue_full", "queue_expired", "queue_paused", "queue_removed", "sensitive_title",
    "not_found", "request_budget_exhausted", "pagination_unverified", "pagination_failed",
    "pagination_loop", "pagination_not_allowed", "config_review_required",
    "candidate_limit_reached", "temporary_failure", "pending_backlog", "robots_disallowed",
})
SAFE_COVERAGE_CODES = frozenset({
    "", "pagination_unverified", "page_limit_reached", "reviewed_range_complete",
    "pagination_failed", "pagination_loop", "pagination_not_allowed",
    "request_budget_exhausted", "config_review_required", "list_check_incomplete",
})
SAFE_RUN_STATES = frozenset({"success", "partial", "failed", "interrupted"})


def _safe_count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _bounded_count(value: Any, limit: int = 10000) -> int:
    return min(limit, _safe_count(value))


def _safe_request_breakdown(raw: Any) -> dict[str, int]:
    raw = raw if isinstance(raw, Mapping) else {}
    return {key: _bounded_count(raw.get(key), 40) for key in ("robots", "list", "detail", "retry")}


def _safe_reasons(raw: Any) -> dict[str, int]:
    reasons: dict[str, int] = {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            count = _safe_count(value)
            if count:
                safe_key = key if key in SAFE_SKIP_REASONS else "other"
                reasons[safe_key] = reasons.get(safe_key, 0) + count
    return reasons


class ManualUpdateRunRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def start(self, *, started_at: datetime | None = None) -> str:
        """Start only AFTER obtaining the OS run lock; close abandoned runs there.

        Task 7 owns that lock. Page reads must use latest(), which never writes.
        """
        timestamp = _iso(started_at or _utc_now())
        run_id = str(uuid.uuid4())
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE update_runs SET state = 'interrupted', finished_at = ? "
                "WHERE finished_at IS NULL", (timestamp,),
            )
            connection.execute(
                "INSERT INTO update_runs(id, started_at, state) VALUES (?, ?, 'running')",
                (run_id, timestamp),
            )
        return run_id

    def finish(
        self, run_id: str, state: str, *, finished_at: datetime | None = None,
    ) -> None:
        safe_state = state if state in SAFE_RUN_STATES else "failed"
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE update_runs SET state = ?, finished_at = ? "
                "WHERE id = ? AND finished_at IS NULL",
                (safe_state, _iso(finished_at or _utc_now()), run_id),
            )

    def latest(self) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM update_runs ORDER BY started_at DESC, rowid DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        if result["finished_at"] is None:
            result["state"] = "interrupted"
        elif result["state"] not in SAFE_RUN_STATES:
            result["state"] = "failed"
        return result


class PendingCandidateRepository:
    """URL-only queue; title admission belongs to the discovery pipeline."""

    def __init__(self, database: Database, sources: Mapping[str, SourceConfig]) -> None:
        self.database = database
        self.sources = dict(sources)

    def _allowed(self, source_id: str, url: str) -> bool:
        source = self.sources.get(source_id)
        return source is not None and evaluate_request(source, url).allowed

    def enqueue(self, source_id: str, canonical_url: str, *, now: datetime | None = None) -> bool:
        if not self._allowed(source_id, canonical_url):
            return False
        timestamp = now or _utc_now()
        with self.database.transaction() as connection:
            self._prune(connection, source_id, timestamp)
            count = connection.execute(
                "SELECT COUNT(*) FROM pending_candidates WHERE source_id = ?", (source_id,),
            ).fetchone()[0]
            if count >= 200:
                return False
            source = self.sources[source_id]
            connection.execute(
                "INSERT OR IGNORE INTO sources(id, name, base_url, created_at) VALUES (?, ?, ?, ?)",
                (source.id, source.name, source.base_url, _iso(timestamp)),
            )
            cursor = connection.execute(
                "INSERT OR IGNORE INTO pending_candidates(source_id, canonical_url, first_seen_at) "
                "VALUES (?, ?, ?)", (source_id, canonical_url, _iso(timestamp)),
            )
            return cursor.rowcount > 0

    def oldest_ready(
        self, source_id: str, *, limit: int = 4, now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        timestamp = now or _utc_now()
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM pending_candidates WHERE source_id = ? AND first_seen_at > ? "
                "AND (paused_until IS NULL OR paused_until <= ?) "
                "ORDER BY first_seen_at ASC, canonical_url ASC",
                (source_id, _iso(timestamp - timedelta(days=90)), _iso(timestamp)),
            ).fetchall()
        return [dict(row) for row in rows if self._allowed(source_id, row["canonical_url"])][
            :max(0, limit)
        ]

    def mark_transient_failure(
        self, source_id: str, canonical_url: str, *, now: datetime | None = None,
    ) -> None:
        timestamp = now or _utc_now()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM pending_candidates WHERE source_id = ? AND canonical_url = ?",
                (source_id, canonical_url),
            ).fetchone()
            if row is None:
                return
            paused_until = _parse_datetime(row["paused_until"])
            if paused_until is not None and paused_until > utc_datetime(timestamp):
                return
            # After a completed pause, count a new consecutive failure streak.
            failures = 1 if paused_until is not None else row["transient_failures"] + 1
            pause = timestamp + timedelta(days=7) if failures >= 3 else None
            connection.execute(
                "UPDATE pending_candidates SET last_attempt_at = ?, transient_failures = ?, "
                "paused_until = ? WHERE source_id = ? AND canonical_url = ?",
                (_iso(timestamp), failures, _iso(pause), source_id, canonical_url),
            )

    def remove(self, source_id: str, canonical_url: str) -> bool:
        """Remove a successful or permanently rejected candidate without storing its reason."""
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM pending_candidates WHERE source_id = ? AND canonical_url = ?",
                (source_id, canonical_url),
            )
            return cursor.rowcount > 0

    def _prune(
        self, connection: sqlite3.Connection, source_id: str | None, now: datetime,
    ) -> int:
        rows = connection.execute(
            "SELECT * FROM pending_candidates" +
            (" WHERE source_id = ?" if source_id is not None else ""),
            (source_id,) if source_id is not None else (),
        ).fetchall()
        removed = 0
        for row in rows:
            if (_parse_datetime(row["first_seen_at"]) <= utc_datetime(now) - timedelta(days=90)
                    or not self._allowed(row["source_id"], row["canonical_url"])):
                connection.execute(
                    "DELETE FROM pending_candidates WHERE source_id = ? AND canonical_url = ?",
                    (row["source_id"], row["canonical_url"]),
                )
                removed += 1
        return removed

    def prune(self, source_id: str | None = None, *, now: datetime | None = None) -> int:
        with self.database.transaction() as connection:
            return self._prune(connection, source_id, now or _utc_now())

    def count(self, source_id: str | None = None) -> int:
        with self.database.connect() as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM pending_candidates" +
                (" WHERE source_id = ?" if source_id is not None else ""),
                (source_id,) if source_id is not None else (),
            ).fetchone()[0])

    def url_only_count(self, source_id: str, visible_urls: Collection[str]) -> int:
        return len(self.urls(source_id) - set(visible_urls))

    def urls(self, source_id: str) -> set[str]:
        """Read all stored URLs, including paused rows, without changing queue state."""
        with self.database.connect() as connection:
            return {row[0] for row in connection.execute(
                "SELECT canonical_url FROM pending_candidates WHERE source_id = ?", (source_id,),
            )}

    def clear(self, source_id: str | None = None) -> int:
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM pending_candidates" +
                (" WHERE source_id = ?" if source_id is not None else ""),
                (source_id,) if source_id is not None else (),
            )
            return cursor.rowcount


class SourceRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def record_status(self, status: Mapping[str, Any]) -> None:
        source_id = str(status["source_id"])
        checked_at = status.get("checked_at", _utc_now())
        if isinstance(checked_at, datetime):
            checked_at = _iso(checked_at)
        raw_state = str(status["state"])
        state = raw_state if raw_state in SAFE_SOURCE_STATES else "error"
        raw_reasons = status.get("skip_reasons", {})
        reasons = _safe_reasons(raw_reasons)
        raw_message = str(status.get("message", ""))
        message = raw_message if raw_message in {
            *SAFE_SOURCE_STATES, *SAFE_TRANSPORT_CODES, *SAFE_SKIP_REASONS,
            *SAFE_COVERAGE_CODES,
            "ok", "robots_not_published"
        } else state
        run_id = status.get("run_id")
        coverage = status.get("coverage_code", "")
        coverage = coverage if coverage in SAFE_COVERAGE_CODES else ""
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO sources(id, name, base_url, created_at) VALUES (?, ?, ?, ?)",
                (
                    source_id,
                    status.get("source_name", source_id),
                    status.get("base_url", ""),
                    checked_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO source_status(
                  source_id, state, checked_at, message,
                  saved_count, updated_count, skipped_count, skip_reasons_json,
                  run_id, list_pages_read, request_count, queued_count, coverage_code,
                  request_breakdown_json, discovered_count, parsed_count,
                  listing_count, pending_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (source_id, state, checked_at, message,
                 _safe_count(status.get("saved_count")), _safe_count(status.get("updated_count")),
                 _safe_count(status.get("skipped_count")), _json(reasons), run_id,
                 _safe_count(status.get("list_pages_read")) if run_id else 0,
                 _safe_count(status.get("request_count")), _safe_count(status.get("queued_count")),
                 coverage, _json(_safe_request_breakdown(status.get("request_breakdown"))),
                 _bounded_count(status.get("discovered_count")),
                 _bounded_count(status.get("parsed_count")),
                 _bounded_count(status.get("listing_count")),
                 _bounded_count(status.get("pending_count"))),
            )

    def latest_statuses(self) -> list[dict[str, Any]]:
        return self._latest_statuses("")

    def latest_successful_list_checks(self) -> list[dict[str, Any]]:
        """Latest real run that parsed a list page, even if details ended partially."""
        return self._latest_statuses(
            "AND candidate.run_id IS NOT NULL AND candidate.list_pages_read > 0 "
            "AND candidate.state != 'synthetic_demo' "
            "AND candidate.message NOT LIKE 'synthetic_demo%'"
        )

    def latest_completed_checks(self) -> list[dict[str, Any]]:
        """Latest historical success; legacy successes retain their NULL run IDs."""
        return self._latest_statuses(
            "AND candidate.state = 'success' AND candidate.message NOT LIKE 'synthetic_demo%'"
        )

    def _latest_statuses(self, history_filter: str) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                  sources.id AS source_id,
                  sources.name AS source_name,
                  source_status.state,
                  source_status.checked_at,
                  source_status.message,
                  source_status.saved_count,
                  source_status.updated_count,
                  source_status.skipped_count,
                  source_status.skip_reasons_json,
                  source_status.run_id,
                  source_status.list_pages_read,
                  source_status.request_count,
                  source_status.queued_count,
                  source_status.coverage_code,
                  source_status.request_breakdown_json,
                  source_status.discovered_count,
                  source_status.parsed_count,
                  source_status.listing_count,
                  source_status.pending_count
                FROM sources
                JOIN source_status ON source_status.id = (
                  SELECT candidate.id
                  FROM source_status AS candidate
                  WHERE candidate.source_id = sources.id
                  {history_filter}
                  ORDER BY candidate.checked_at DESC, candidate.id DESC
                  LIMIT 1
                )
                ORDER BY sources.id
                """
            ).fetchall()
        statuses = []
        for row in rows:
            status = dict(row)
            offline_seed = (
                status["state"] == "synthetic_demo"
                or str(status["message"]).startswith("synthetic_demo")
            )
            if offline_seed:
                status["state"] = "not_checked"
                status["checked_at"] = None
                status["message"] = ""
            elif status["state"] not in SAFE_SOURCE_STATES:
                status["state"] = "error"
            if not offline_seed and status["message"] not in {
                *SAFE_SOURCE_STATES, *SAFE_TRANSPORT_CODES, *SAFE_SKIP_REASONS,
                *SAFE_COVERAGE_CODES,
                "ok", "robots_not_published"
            }:
                status["message"] = status["state"]
            try:
                raw_reasons = json.loads(status.pop("skip_reasons_json"))
            except (TypeError, json.JSONDecodeError):
                raw_reasons = {}
            status["skip_reasons"] = {
                key: value for key, value in raw_reasons.items()
                if key in SAFE_SKIP_REASONS and isinstance(value, int) and value > 0
            } if isinstance(raw_reasons, dict) else {}
            try:
                raw_breakdown = json.loads(status.pop("request_breakdown_json"))
            except (TypeError, json.JSONDecodeError):
                raw_breakdown = {}
            status["request_breakdown"] = _safe_request_breakdown(raw_breakdown)
            for field in ("discovered_count", "parsed_count", "listing_count", "pending_count"):
                status[field] = _bounded_count(status[field])
            if status["coverage_code"] not in SAFE_COVERAGE_CODES:
                status["coverage_code"] = ""
            for field in ("saved_count", "updated_count", "skipped_count", "list_pages_read",
                          "request_count", "queued_count"):
                status[field] = _safe_count(status[field])
            if offline_seed or status["run_id"] is None:
                status["list_pages_read"] = 0
            statuses.append(status)
        return statuses


class RecommendationRunRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def save(self, run: Mapping[str, Any]) -> str:
        run_id = str(run.get("id") or uuid.uuid4())
        created_at = run.get("created_at", _utc_now())
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO recommendation_runs(id, persona_id, parameters_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, run.get("persona_id"), _json(run.get("parameters", {})), created_at),
            )
            for rank, item in enumerate(run.get("items", ()), start=1):
                connection.execute(
                    """
                    INSERT INTO recommendation_items(
                      run_id, notice_id, rank, score, breakdown_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        item["notice_id"],
                        int(item.get("rank", rank)),
                        float(item["score"]),
                        _json(item.get("breakdown", {})),
                    ),
                )
        return run_id

    def summary(self) -> dict[str, int | str | None]:
        with self.database.connect() as connection:
            run_row = connection.execute(
                """
                SELECT COUNT(*) AS run_count, MAX(created_at) AS latest_at
                FROM recommendation_runs
                """
            ).fetchone()
            item_count = int(
                connection.execute("SELECT COUNT(*) FROM recommendation_items").fetchone()[0]
            )
        return {
            "run_count": int(run_row["run_count"]),
            "item_count": item_count,
            "latest_at": run_row["latest_at"],
        }
