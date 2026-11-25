from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import (
    NoticeRepository,
    PersonaRepository,
    RecommendationRunRepository,
    SourceRepository,
)
from shu_insight.db.schema import Database
from shu_insight.domain.models import (
    DeadlineConfidence,
    DiscoveredNotice,
    NoticeDraft,
    ParseState,
    Persona,
    ProofStatus,
    Verification,
)
from shu_insight.sources.registry import load_source_registry


def test_empty_parsed_summaries_are_excluded_from_mining(database, notice_draft):
    repo = NoticeRepository(database)
    empty = repo.upsert(replace(notice_draft, normalized_text=""), summary_override="")
    populated = repo.upsert(replace(
        notice_draft, canonical_url=notice_draft.canonical_url + "?v=2",
    ))
    repo.refresh_mining()
    with database.connect() as connection:
        ids = {row[0] for row in connection.execute(
            "SELECT notice_id FROM duplicate_cluster_members",
        )}
    assert empty.notice_id not in ids
    assert populated.notice_id in ids


def test_schema_v1_migrates_version_history_to_allow_repeated_fingerprints(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v1.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_meta (version INTEGER NOT NULL);
            INSERT INTO schema_meta VALUES (1);
            CREATE TABLE notice_versions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              notice_id TEXT NOT NULL,
              content_fingerprint TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(notice_id, content_fingerprint)
            );
            """
        )

    database = Database(path)
    database.initialize()

    with database.connect() as connection:
        assert connection.execute(
            "SELECT version FROM schema_meta"
        ).fetchone()[0] == 5
        unique_indexes = [
            row
            for row in connection.execute("PRAGMA index_list(notice_versions)")
            if row[2]
        ]
    assert unique_indexes == []


def test_interrupted_v1_migration_can_resume_without_losing_history(
    tmp_path: Path,
    notice_draft: NoticeDraft,
) -> None:
    path = tmp_path / "partial-migration.sqlite3"
    database = Database(path)
    database.initialize()
    NoticeRepository(database).upsert(notice_draft)
    with database.connect() as connection:
        original_hash = connection.execute(
            "SELECT content_fingerprint FROM notice_versions"
        ).fetchone()[0]
        connection.execute("ALTER TABLE notice_versions RENAME TO notice_versions_v1")
        connection.execute("UPDATE schema_meta SET version = 1")

    database.initialize()
    database.initialize()

    with database.connect() as connection:
        assert connection.execute("SELECT version FROM schema_meta").fetchone()[0] == 5
        assert connection.execute(
            "SELECT content_fingerprint FROM notice_versions"
        ).fetchone()[0] == original_hash
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'notice_versions_v1'"
        ).fetchone() is None


@pytest.fixture
def database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "test.sqlite3")
    database.initialize()
    return database


@pytest.fixture
def notice_draft() -> NoticeDraft:
    return NoticeDraft(
        source_id="graduate",
        canonical_url="https://gs.shu.edu.cn/info/1029/173242.htm",
        title="上海大学2026级硕博连读研究生、直博生 申请课程免修及成绩认定通知",
        normalized_text=(
            "上海大学研究生院通知，2026级硕博连读研究生和直博生可按条件申请课程免修或"
            "学分、成绩认定。学生需先完成个人培养计划，再于2026年9月11日至10月11日"
            "在研究生综合服务平台申请，经导师、学院和研究生院审核。"
        ),
        published_at=datetime(2026, 9, 11, 5, 50, tzinfo=UTC),
        # Date-only application end; midnight UTC is an anchor, not a claimed hour.
        deadline_at=datetime(2026, 10, 11, tzinfo=UTC),
        deadline_confidence=DeadlineConfidence.LOW,
        department="研究生院",
        column_name="培养管理",
    )


def _official_policy() -> OfficialNoticePolicy:
    registry = load_source_registry(Path("config/sources.toml"))
    return OfficialNoticePolicy({key: item.config for key, item in registry.items()})


def test_listing_evidence_is_browseable_but_not_parsed(database, notice_draft):
    repo = NoticeRepository(database, policy=_official_policy())
    checked_at = datetime(2026, 10, 19, tzinfo=UTC)
    list_url = "https://gs.shu.edu.cn/xwlb/sy.htm"
    item = DiscoveredNotice(notice_draft.source_id, notice_draft.canonical_url,
                            notice_draft.title, "2026-09-11")
    result = repo.upsert_listing(item, list_url=list_url, checked_at=checked_at)
    record, = repo.list_browseable()
    assert result.created and not result.updated
    assert record.id == result.notice_id
    assert record.summary == ""
    assert record.parse_state == ParseState.LISTING_ONLY
    assert record.proof_status == ProofStatus.VERIFIED_LISTING
    assert record.verified_at == record.fetched_at == checked_at
    assert record.verification_ref == "listing:" + hashlib.sha256(list_url.encode()).hexdigest()
    assert record.published_at == datetime(2026, 9, 10, 16, tzinfo=UTC)
    assert record.deadline_at is None
    assert record.deadline_confidence == DeadlineConfidence.UNKNOWN
    assert repo.list_verified() == []
    assert repo.detail_evidence_verified(result.notice_id) == {
        "duplicates": [], "possible_duplicates": [], "versions": [],
    }
    assert repo.upsert_listing(item, list_url=list_url, checked_at=checked_at) == replace(
        result, created=False,
    )
    proof = Verification(ProofStatus.VERIFIED_LIVE, checked_at, "live:fixture")
    upgrade = repo.upsert(notice_draft, verification=proof)
    assert upgrade.notice_id == result.notice_id and upgrade.updated
    assert repo.list_verified()[0].summary
    assert len(repo.list_browseable()) == 1


@pytest.mark.parametrize("published_hint", [None, "", "not a date", "2026-02-30"])
def test_listing_without_date_stays_unknown(database, notice_draft, published_hint):
    repo = NoticeRepository(database, policy=_official_policy())
    repo.upsert_listing(DiscoveredNotice(
        notice_draft.source_id, notice_draft.canonical_url, notice_draft.title, published_hint,
    ), list_url="https://gs.shu.edu.cn/xwlb/sy.htm",
        checked_at=datetime(2026, 10, 19, tzinfo=UTC))
    record, = repo.list_browseable()
    assert record.published_at is None and record.deadline_at is None


@pytest.mark.parametrize("status", [ProofStatus.VERIFIED_LIVE, ProofStatus.VERIFIED_SNAPSHOT])
def test_listing_cannot_downgrade_parsed_notice(database, notice_draft, status):
    repo = NoticeRepository(database, policy=_official_policy())
    proof = Verification(status, datetime(2026, 10, 18, tzinfo=UTC), "live:fixture")
    original = repo.upsert(notice_draft, verification=proof)
    before = repo.list_verified()[0]
    result = repo.upsert_listing(DiscoveredNotice(
        notice_draft.source_id, notice_draft.canonical_url, "列表中的不同标题",
    ), list_url="https://gs.shu.edu.cn/xwlb/sy.htm",
        checked_at=datetime(2026, 10, 19, tzinfo=UTC))
    assert result.notice_id == original.notice_id and not result.updated
    assert repo.list_browseable() == [before]
    assert not repo.withdraw_listing(notice_draft.canonical_url)
    assert repo.list_verified() == [before]


def test_listing_promotion_clears_all_legacy_body_metadata(database, notice_draft):
    repo = NoticeRepository(database, policy=_official_policy())
    original = repo.upsert(replace(
        notice_draft, location="旧地点", department="旧部门", column_name="旧栏目",
        education_levels=("硕士生",), grades=("研一",), colleges=("旧学院",),
        majors=("旧专业",), categories=("旧分类",), topic_tags=("旧标签",),
    ))

    saved = repo.upsert_listing(
        DiscoveredNotice("graduate", notice_draft.canonical_url, "课程公开通知", "2026-09-21"),
        list_url="https://gs.shu.edu.cn/xwlb/sy.htm",
        checked_at=datetime(2026, 10, 19, tzinfo=UTC),
    )

    record, = repo.list_browseable()
    assert saved.notice_id == original.notice_id == record.id
    assert saved.updated and not saved.created
    assert repo.count() == 1
    assert record.parse_state == ParseState.LISTING_ONLY
    assert record.proof_status == ProofStatus.VERIFIED_LISTING
    assert record.title == "课程公开通知"
    assert record.published_at == datetime(2026, 9, 20, 16, tzinfo=UTC)
    assert record.summary == record.location == record.department == record.column_name == ""
    assert record.deadline_at is None
    assert record.deadline_confidence == DeadlineConfidence.UNKNOWN
    assert record.education_levels == record.grades == record.colleges == record.majors == ()
    assert record.categories == record.topic_tags == ()


def test_attachment_tombstone_is_not_relisted_on_second_click(database, notice_draft):
    repo = NoticeRepository(database, policy=_official_policy())
    item = DiscoveredNotice(notice_draft.source_id, notice_draft.canonical_url, notice_draft.title)
    list_url = "https://gs.shu.edu.cn/xwlb/sy.htm"
    result = repo.upsert_listing(item, list_url=list_url,
                                 checked_at=datetime(2026, 10, 18, tzinfo=UTC))
    assert repo.withdraw_listing(item.url)
    assert not repo.withdraw_listing(item.url)
    assert not repo.withdraw_listing("https://gs.shu.edu.cn/info/1/missing.htm")
    withdrawn = repo.find_by_url(item.url)
    assert withdrawn.proof_status == ProofStatus.ATTACHMENT_EXCLUDED
    assert withdrawn.verification_ref == "" and withdrawn.verified_at is None
    replay = repo.upsert_listing(replace(item, published_hint="2026-10-19"),
                                list_url=list_url,
                                checked_at=datetime(2026, 10, 19, tzinfo=UTC))
    assert replay.notice_id == result.notice_id and not replay.updated
    assert repo.list_browseable() == repo.list_verified() == []
    assert repo.count() == 1
    changed = repo.upsert_listing(replace(item, title="已更新的通知标题"), list_url=list_url,
                                 checked_at=datetime(2026, 10, 19, tzinfo=UTC))
    assert changed.updated and changed.notice_id == result.notice_id
    assert len(repo.list_browseable()) == 1


@pytest.mark.parametrize(("list_url", "url", "title"), [
    ("https://gs.shu.edu.cn/info/1/2.htm", "https://gs.shu.edu.cn/info/1/3.htm", "通知"),
    ("https://gs.shu.edu.cn/xwlb/unreviewed.htm", "https://gs.shu.edu.cn/info/1/3.htm", "通知"),
    ("https://evil.example/xwlb/sy.htm", "https://gs.shu.edu.cn/info/1/3.htm", "通知"),
    ("https://gs.shu.edu.cn/xwlb/sy.htm", "https://evil.example/info/1/3.htm", "通知"),
    ("https://gs.shu.edu.cn/xwlb/sy.htm", "https://gs.shu.edu.cn/info/file.pdf", "通知"),
    ("https://gs.shu.edu.cn/xwlb/sy.htm", "https://gs.shu.edu.cn/info/1/3.htm", ""),
    ("https://gs.shu.edu.cn/xwlb/sy.htm", "https://gs.shu.edu.cn/info/1/3.htm", "[演示]虚构通知"),
])
def test_url_only_or_off_host_listing_is_not_browseable(database, list_url, url, title):
    repo = NoticeRepository(database, policy=_official_policy())
    with pytest.raises(ValueError):
        repo.upsert_listing(DiscoveredNotice("graduate", url, title), list_url=list_url,
                            checked_at=datetime(2026, 10, 19, tzinfo=UTC))
    assert repo.list_browseable() == [] and repo.count() == 0


def test_listing_requires_policy_and_aware_evidence_time(database, notice_draft):
    item = DiscoveredNotice(notice_draft.source_id, notice_draft.canonical_url, notice_draft.title)
    with pytest.raises(RuntimeError, match="policy"):
        NoticeRepository(database).upsert_listing(item, list_url="https://gs.shu.edu.cn/xwlb/sy.htm",
                                                 checked_at=datetime(2026, 10, 19, tzinfo=UTC))
    with pytest.raises(RuntimeError, match="policy"):
        NoticeRepository(database).list_browseable()
    with pytest.raises(ValueError, match="aware"):
        NoticeRepository(database, policy=_official_policy()).upsert_listing(
            item, list_url="https://gs.shu.edu.cn/xwlb/sy.htm", checked_at=datetime(2026, 10, 19),
        )


def test_listing_does_not_create_body_duplicate_cluster(database, notice_draft):
    repo = NoticeRepository(database, policy=_official_policy())
    for url in (notice_draft.canonical_url, "https://gs.shu.edu.cn/info/1029/173243.htm"):
        repo.upsert_listing(DiscoveredNotice("graduate", url, notice_draft.title),
                            list_url="https://gs.shu.edu.cn/xwlb/sy.htm",
                            checked_at=datetime(2026, 10, 19, tzinfo=UTC))
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM duplicate_cluster_members",
        ).fetchone()[0] == 0
    repo.withdraw_listing(notice_draft.canonical_url)
    repo.refresh_mining()
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM duplicate_cluster_members",
        ).fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM duplicate_relations").fetchone()[0] == 0


@pytest.mark.parametrize("operation", ["reads", "recheck", "write", "refresh_304"])
def test_narrowed_current_scope_rejects_existing_verified_article(
    database, notice_draft, operation,
):
    repo = NoticeRepository(database, policy=_official_policy())
    proof = Verification(ProofStatus.VERIFIED_LIVE, datetime(2026, 10, 6, tzinfo=UTC),
                         "live:fixture")
    notice_id = repo.upsert(notice_draft, verification=proof).notice_id
    source = repo.policy.sources["graduate"]
    repo.policy = OfficialNoticePolicy({"graduate": replace(
        source, allowed_path_prefixes=("/xwlb/sy", "/info/1029/500.htm"),
    )})
    if operation == "reads":
        assert repo.list_verified() == []
        assert repo.detail_evidence_verified(notice_id) == {
            "duplicates": [], "possible_duplicates": [], "versions": [],
        }
    elif operation == "recheck":
        assert repo.list_live_for_recheck("graduate", set()) == []
    elif operation == "write":
        with pytest.raises(ValueError, match="official article policy"):
            repo.upsert(notice_draft, verification=proof)
    else:
        assert not repo.refresh_live_verification(
            notice_draft.canonical_url, datetime(2026, 10, 7, tzinfo=UTC),
        )
    assert repo.count() == 1
    assert repo.find_by_url(notice_draft.canonical_url).verified_at == proof.verified_at


def test_verified_read_hides_unproved_and_demo_notices(
    database: Database, notice_draft: NoticeDraft,
) -> None:
    repo = NoticeRepository(database, policy=_official_policy())
    proof = Verification(
        ProofStatus.VERIFIED_SNAPSHOT, datetime(2026, 9, 24, tzinfo=UTC),
        "snapshot:entry-1",
    )
    verified_id = repo.upsert(notice_draft, verification=proof).notice_id
    repo.upsert(NoticeDraft(
        source_id="computer", canonical_url="https://cs.shu.edu.cn/info/1/legacy.htm",
        title="旧通知", normalized_text="旧通知内容",
    ))
    repo.upsert(NoticeDraft(
        source_id="computer", canonical_url="https://demo.invalid/computer/demo-1",
        title="[演示]通知", normalized_text="演示数据", topic_tags=("synthetic_demo",),
    ))
    assert len(repo.list_all()) == 3
    assert [row.id for row in repo.list_verified()] == [verified_id]
    assert (repo.list_verified()[0].canonical_url, repo.list_verified()[0].title) == (
        "https://gs.shu.edu.cn/info/1029/173242.htm",
        "上海大学2026级硕博连读研究生、直博生 申请课程免修及成绩认定通知",
    )
    assert repo.list_verified()[0].deadline_at.date().isoformat() == "2026-10-11"
    assert repo.list_verified()[0].deadline_confidence == DeadlineConfidence.LOW


def test_verified_detail_hides_legacy_before_values_but_keeps_verified_changes(
    database: Database,
) -> None:
    manifest = json.loads(Path("data/official/snapshot-2026-09-24.json").read_text(
        encoding="utf-8"))
    entry = next(row for row in manifest["entries"] if row["id"] == "shu-main-400245")
    repo = NoticeRepository(database, policy=_official_policy())
    url = entry["canonical_url"]
    legacy_title = "UNVERIFIED_LEGACY_TITLE"
    legacy_summary = "UNVERIFIED_LEGACY_SUMMARY"
    notice_id = repo.upsert(NoticeDraft(
        source_id="shu_main", canonical_url=url, title=legacy_title,
        normalized_text=legacy_summary,
    )).notice_id
    verified_draft = NoticeDraft(
        source_id="shu_main", canonical_url=url, title=entry["title"],
        normalized_text=entry["summary"],
        published_at=datetime.fromisoformat(entry["published_at"] + "T00:00:00+08:00"),
    )
    proof = Verification(ProofStatus.VERIFIED_SNAPSHOT,
                         datetime.fromisoformat(entry["verified_at"]),
                         "snapshot:shu-main-400245")
    repo.upsert(verified_draft, verification=proof, summary_override=entry["summary"])
    raw = repo.detail_evidence(notice_id)
    assert legacy_title in json.dumps(raw, ensure_ascii=False)
    assert legacy_summary in json.dumps(raw, ensure_ascii=False)
    first_verified = repo.detail_evidence_verified(notice_id)
    assert legacy_title not in json.dumps(first_verified, ensure_ascii=False)
    assert legacy_summary not in json.dumps(first_verified, ensure_ascii=False)

    repo.upsert(
        replace(verified_draft,
                deadline_at=datetime.fromisoformat(entry["deadline_at"]),
                deadline_confidence=DeadlineConfidence.HIGH),
        verification=proof, summary_override=entry["summary"],
    )
    safe_versions = repo.detail_evidence_verified(notice_id)["versions"]
    assert len(safe_versions) == 2
    deadline_utc = datetime.fromisoformat(entry["deadline_at"]).astimezone(UTC).isoformat()
    assert any(change["field"] == "deadline_at" and change["before"] is None
               and change["after"] == deadline_utc
               for change in safe_versions[-1]["changes"])


def test_verified_read_rejects_naive_timestamp_stored_in_database(
    database: Database, notice_draft: NoticeDraft,
) -> None:
    repo = NoticeRepository(database, policy=_official_policy())
    proof = Verification(
        ProofStatus.VERIFIED_SNAPSHOT, datetime(2026, 9, 24, tzinfo=UTC),
        "snapshot:entry-1",
    )
    notice_id = repo.upsert(notice_draft, verification=proof).notice_id
    with database.connect() as connection:
        connection.execute(
            "UPDATE notices SET verified_at = ? WHERE id = ?",
            ("2026-09-24T00:00:00", notice_id),
        )
    assert repo.list_verified() == []


@pytest.mark.parametrize("proof", [
    Verification(ProofStatus.UNVERIFIED_LEGACY, datetime(2026, 9, 24, tzinfo=UTC), "x"),
    Verification(ProofStatus.VERIFIED_SNAPSHOT, datetime(2026, 9, 24), "x"),
    Verification(ProofStatus.VERIFIED_SNAPSHOT, datetime(2026, 9, 24, tzinfo=UTC), " "),
])
def test_upsert_rejects_invalid_verification(
    database: Database, notice_draft: NoticeDraft, proof: Verification,
) -> None:
    repo = NoticeRepository(database, policy=_official_policy())
    with pytest.raises(ValueError):
        repo.upsert(notice_draft, verification=proof)
    assert repo.count() == 0


def test_verified_upsert_rejects_source_url_mismatch(
    database: Database, notice_draft: NoticeDraft,
) -> None:
    repo = NoticeRepository(database, policy=_official_policy())
    proof = Verification(
        ProofStatus.VERIFIED_SNAPSHOT, datetime(2026, 9, 24, tzinfo=UTC), "snapshot:entry-1"
    )
    with pytest.raises(ValueError):
        repo.upsert(NoticeDraft(
            source_id="computer", canonical_url=notice_draft.canonical_url,
            title=notice_draft.title, normalized_text=notice_draft.normalized_text,
        ), verification=proof)
    assert repo.count() == 0


def test_same_content_reverification_updates_time_without_new_version(
    database: Database, notice_draft: NoticeDraft,
) -> None:
    repo = NoticeRepository(database, policy=_official_policy())
    first = Verification(
        ProofStatus.VERIFIED_SNAPSHOT, datetime(2026, 9, 24, tzinfo=UTC), "snapshot:entry-1"
    )
    later = Verification(
        ProofStatus.VERIFIED_LIVE, datetime(2026, 9, 25, tzinfo=UTC), "live:hash-1"
    )
    notice_id = repo.upsert(notice_draft, verification=first).notice_id
    before = len(repo.detail_evidence_verified(notice_id)["versions"])
    repo.upsert(notice_draft, verification=later)
    record = repo.find_by_url(notice_draft.canonical_url)
    assert record is not None
    assert record.proof_status == ProofStatus.VERIFIED_LIVE
    assert record.verified_at == later.verified_at
    assert record.verification_ref == later.reference
    assert len(repo.detail_evidence_verified(notice_id)["versions"]) == before


def test_verified_evidence_hides_old_unproved_payload_and_unproved_relation(
    database: Database, notice_draft: NoticeDraft,
) -> None:
    repo = NoticeRepository(database, policy=_official_policy())
    main_id = repo.upsert(notice_draft).notice_id
    other = NoticeDraft(
        source_id="computer", canonical_url="https://cs.shu.edu.cn/info/1/other.htm",
        title="相似通知", normalized_text=notice_draft.normalized_text,
    )
    other_id = repo.upsert(other).notice_id
    proof = Verification(
        ProofStatus.VERIFIED_SNAPSHOT, datetime(2026, 9, 24, tzinfo=UTC), "snapshot:entry-1"
    )
    repo.upsert(notice_draft, verification=proof)
    with database.connect() as connection:
        main = connection.execute(
            "SELECT content_fingerprint FROM notices WHERE id = ?", (main_id,)
        ).fetchone()
        relation = connection.execute(
            "SELECT COUNT(*) FROM duplicate_relations WHERE left_id = ? OR right_id = ?",
            (main_id, main_id),
        ).fetchone()[0]
        if not relation:
            other_row = connection.execute(
                "SELECT content_fingerprint FROM notices WHERE id = ?", (other_id,)
            ).fetchone()
            connection.execute(
                "INSERT INTO duplicate_relations VALUES (?, ?, ?, ?, ?, ?)",
                (main_id, other_id, main[0], other_row[0],
                 '{"disposition":"duplicate"}', "2026-09-24T00:00:00+00:00"),
            )
    evidence = repo.detail_evidence_verified(main_id)
    assert evidence["duplicates"] == []
    assert evidence["possible_duplicates"] == []
    assert evidence["versions"] == []
    assert repo.detail_evidence_verified(other_id) == {
        "duplicates": [], "possible_duplicates": [], "versions": []
    }


def test_summary_override_is_preserved_for_verified_record(
    database: Database, notice_draft: NoticeDraft,
) -> None:
    repo = NoticeRepository(database, policy=_official_policy())
    proof = Verification(
        ProofStatus.VERIFIED_SNAPSHOT, datetime(2026, 9, 24, tzinfo=UTC), "snapshot:entry-1"
    )
    summary = (
        "上海大学研究生院面向2026级硕博连读研究生、直博生发布课程免修及成绩认定安排。"
        "硕博连读研究生在硕士阶段修过且成绩合格的素质课，如学时学分达到博士培养方案"
        "同类课程要求，可申请同类课程免修；此前修过的博士课程如满足内容和学时学分条件，"
        "可申请博士阶段学分成绩认定。直博生在本科阶段修读的研究生课程符合相应条件时，"
        "也可申请认定。学生完成个人培养计划后在研究生综合服务平台申请，"
        "经导师、学院和研究生院审核；申请时间为2026年9月11日至10月11日。"
    )
    repo.upsert(notice_draft, verification=proof, summary_override=summary)
    record = repo.list_verified()[0]
    assert record.summary == summary
    assert len(record.summary) > 160


def test_database_enables_foreign_keys(database: Database) -> None:
    with database.connect() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_notice_upsert_is_idempotent(database: Database, notice_draft: NoticeDraft) -> None:
    repo = NoticeRepository(database)

    first = repo.upsert(notice_draft)
    second = repo.upsert(notice_draft)

    assert first.created is True
    assert second.created is False
    assert second.notice_id == first.notice_id
    assert repo.count() == 1
    assert len(repo.list_all()) == 1


def test_delete_all_personas_removes_preferences(database: Database) -> None:
    repo = PersonaRepository(database)
    persona = Persona(
        id="demo",
        label="大二保研科研",
        goals=("保研", "科研"),
        interest_tags=("竞赛",),
        custom_keywords=("推免",),
    )

    repo.save(persona)
    repo.delete_all()

    assert repo.list_all() == []


def test_notice_and_version_rollback_together(
    database: Database, notice_draft: NoticeDraft, monkeypatch
) -> None:
    repo = NoticeRepository(database)

    def fail_version_insert(*_args, **_kwargs) -> None:
        raise sqlite3.IntegrityError("forced version failure")

    monkeypatch.setattr(repo, "_insert_version", fail_version_insert)

    with pytest.raises(sqlite3.IntegrityError, match="forced version failure"):
        repo.upsert(notice_draft)

    assert repo.count() == 0
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM notice_versions").fetchone()[0] == 0


def test_source_repository_returns_only_latest_status(database: Database) -> None:
    repo = SourceRepository(database)
    repo.record_status(
        {
            "source_id": "computer",
            "source_name": "计算机学院",
            "state": "error",
            "checked_at": datetime(2026, 9, 20, tzinfo=UTC),
            "message": "temporary failure",
        }
    )
    repo.record_status(
        {
            "source_id": "computer",
            "source_name": "计算机学院",
            "state": "success",
            "checked_at": datetime(2026, 9, 21, tzinfo=UTC),
            "message": "ok",
        }
    )

    statuses = repo.latest_statuses()

    assert statuses == [
        {
            "source_id": "computer",
            "source_name": "计算机学院",
            "state": "success",
            "checked_at": "2026-09-21T00:00:00+00:00",
            "message": "ok",
            "saved_count": 0,
            "updated_count": 0,
            "skipped_count": 0,
            "skip_reasons": {},
            "run_id": None,
            "list_pages_read": 0,
            "request_count": 0,
            "queued_count": 0,
            "coverage_code": "",
            "discovered_count": 0,
            "parsed_count": 0,
            "listing_count": 0,
            "pending_count": 0,
            "request_breakdown": {"robots": 0, "list": 0, "detail": 0, "retry": 0},
        }
    ]


def test_existing_source_status_exception_is_sanitized_on_read(database: Database) -> None:
    SourceRepository(database).record_status({"source_id": "computer", "state": "error"})
    with database.transaction() as connection:
        connection.execute(
            "UPDATE source_status SET message = ? WHERE source_id = ?",
            ("<script>https://example.invalid/?phone=13800138000</script>", "computer"),
        )
    status = SourceRepository(database).latest_statuses()[0]
    assert status["message"] == "error"


def test_legacy_offline_success_is_not_reported_as_website_access(database: Database) -> None:
    repo = SourceRepository(database)
    repo.record_status({
        "source_id": "computer", "state": "success",
        "checked_at": datetime(2026, 9, 20, tzinfo=UTC), "message": "ok",
    })
    with database.transaction() as connection:
        connection.execute(
            "UPDATE source_status SET message = ? WHERE source_id = ?",
            ("synthetic_demo：离线种子，无网络请求", "computer"),
        )
    legacy = repo.latest_statuses()[0]
    assert legacy["state"] == "not_checked"
    assert legacy["checked_at"] is None
    assert legacy["message"] == ""
    repo.record_status({
        "source_id": "computer", "state": "success",
        "checked_at": datetime(2026, 9, 21, tzinfo=UTC), "message": "ok",
    })
    assert repo.latest_statuses()[0]["state"] == "success"


def test_recommendation_repository_summarizes_saved_runs(
    database: Database,
    notice_draft: NoticeDraft,
) -> None:
    notice_id = NoticeRepository(database).upsert(notice_draft).notice_id
    repo = RecommendationRunRepository(database)
    repo.save(
        {
            "id": "run-1",
            "created_at": datetime(2026, 9, 21, tzinfo=UTC),
            "items": [{"notice_id": notice_id, "score": 0.8}],
        }
    )

    assert repo.summary() == {
        "run_count": 1,
        "item_count": 1,
        "latest_at": "2026-09-21T00:00:00+00:00",
    }
