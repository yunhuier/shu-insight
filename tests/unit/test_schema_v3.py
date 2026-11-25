from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shu_insight.db.schema import Database


def make_v2_database(tmp_path: Path) -> Path:
    path = tmp_path / "legacy.sqlite3"
    schema = Path("tests/fixtures/db/schema_v2.sql").read_text(encoding="utf-8")
    with sqlite3.connect(path) as db:
        db.executescript(schema)
        db.execute("INSERT INTO schema_meta(version) VALUES (2)")
        db.execute(
            "INSERT INTO sources VALUES ('computer','computer','https://cs.shu.edu.cn/','2026-09-21')"
        )
        db.execute("INSERT INTO personas VALUES ('kept', 'kept', '{}', '2026-09-21')")
        db.execute(
            "INSERT INTO notices(id,canonical_url,source_id,title,deadline_confidence,summary,"
            "content_fingerprint,summary_simhash,parse_state,fetched_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "demo-id",
                "https://demo.invalid/x",
                "computer",
                "[演示] 测试",
                "unknown",
                "测试",
                "hash",
                0,
                "parsed",
                "2026-09-21T00:00:00+00:00",
            ),
        )
        db.execute(
            "INSERT INTO source_status(source_id,state,checked_at) "
            "VALUES ('computer','success','2026-09-21')"
        )
    return path


def test_v2_upgrade_backs_up_and_preserves_legacy(tmp_path: Path) -> None:
    path = make_v2_database(tmp_path)

    Database(path).initialize()

    with sqlite3.connect(path) as db:
        assert db.execute("SELECT version FROM schema_meta").fetchone()[0] == 5
        assert db.execute("SELECT payload_json FROM personas WHERE id='kept'").fetchone()[0] == "{}"
        assert db.execute(
            "SELECT canonical_url,proof_status,verified_at,verification_ref FROM notices"
        ).fetchone() == ("https://demo.invalid/x", "unverified_legacy", None, "")
        assert db.execute(
            "SELECT saved_count,updated_count,skipped_count,skip_reasons_json FROM source_status"
        ).fetchone() == (0, 0, 0, "{}")
    backups = list((tmp_path / "backups").glob("*.sqlite3"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert backup.execute("SELECT version FROM schema_meta").fetchone()[0] == 2
        assert backup.execute("SELECT COUNT(*) FROM notices").fetchone()[0] == 1


def test_backup_failure_leaves_v2_unchanged(tmp_path: Path, monkeypatch) -> None:
    path = make_v2_database(tmp_path)
    monkeypatch.setattr(
        Database,
        "_backup_before_migration",
        lambda *_: (_ for _ in ()).throw(OSError("backup failed")),
    )

    with pytest.raises(OSError, match="backup failed"):
        Database(path).initialize()

    with sqlite3.connect(path) as db:
        assert db.execute("SELECT version FROM schema_meta").fetchone()[0] == 2
        assert db.execute("SELECT payload_json FROM personas WHERE id='kept'").fetchone()
        assert "proof_status" not in {col[1] for col in db.execute("PRAGMA table_info(notices)")}


def test_failed_migration_rolls_back_after_a_verified_backup(tmp_path: Path, monkeypatch) -> None:
    path = make_v2_database(tmp_path)

    def fail_after_first_ddl(_self, connection: sqlite3.Connection) -> None:
        connection.execute("ALTER TABLE notices ADD COLUMN proof_status TEXT")
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(Database, "_migrate_v2_to_v3", fail_after_first_ddl)
    with pytest.raises(RuntimeError, match="injected migration failure"):
        Database(path).initialize()

    with sqlite3.connect(path) as db:
        assert db.execute("SELECT version FROM schema_meta").fetchone()[0] == 2
        assert "proof_status" not in {col[1] for col in db.execute("PRAGMA table_info(notices)")}
        assert db.execute("SELECT payload_json FROM personas WHERE id='kept'").fetchone()
    assert len(list((tmp_path / "backups").glob("*.sqlite3"))) == 1


def test_failed_v1_upgrade_rolls_back_all_schema_changes(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "v1.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript(
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
            CREATE TABLE personas (
              id TEXT PRIMARY KEY,
              label TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            INSERT INTO personas VALUES ('kept', 'kept', '{}', '2026-09-21');
            """
        )

    def fail_during_v3(_self, connection: sqlite3.Connection) -> None:
        connection.execute("ALTER TABLE notices ADD COLUMN extra TEXT")
        raise RuntimeError("injected v3 failure")

    monkeypatch.setattr(Database, "_migrate_v2_to_v3", fail_during_v3)
    with pytest.raises(RuntimeError, match="injected v3 failure"):
        Database(path).initialize()

    with sqlite3.connect(path) as db:
        assert db.execute("SELECT version FROM schema_meta").fetchone()[0] == 1
        assert {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            if not row[0].startswith("sqlite_")
        } == {"schema_meta", "notice_versions", "personas"}
        assert db.execute(
            "SELECT payload_json FROM personas WHERE id='kept'"
        ).fetchone()[0] == "{}"
        assert any(row[2] for row in db.execute("PRAGMA index_list(notice_versions)"))
    backups = list((tmp_path / "backups").glob("*.sqlite3"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert backup.execute("SELECT version FROM schema_meta").fetchone()[0] == 1


def test_new_database_has_v3_schema_without_backup(tmp_path: Path) -> None:
    path = tmp_path / "new.sqlite3"

    Database(path).initialize()

    with sqlite3.connect(path) as db:
        assert db.execute("SELECT version FROM schema_meta").fetchone()[0] == 5
        assert "proof_status" in {col[1] for col in db.execute("PRAGMA table_info(notices)")}
        assert "skip_reasons_json" in {
            col[1] for col in db.execute("PRAGMA table_info(source_status)")
        }
    assert not (tmp_path / "backups").exists()


def test_verification_model_carries_status_time_and_reference() -> None:
    from shu_insight.domain.models import (
        DeadlineConfidence,
        NoticeRecord,
        ProofStatus,
        Verification,
    )

    when = datetime(2026, 9, 25, tzinfo=UTC)
    proof = Verification(ProofStatus.VERIFIED_SNAPSHOT, when, "archive:sha256:abc")

    assert proof.status == ProofStatus.VERIFIED_SNAPSHOT
    assert proof.verified_at == when
    assert proof.reference == "archive:sha256:abc"
    assert ProofStatus.VERIFIED_LIVE == "verified_live"
    legacy = NoticeRecord(
        id="old",
        source_id="computer",
        canonical_url="https://demo.invalid/x",
        title="legacy",
        summary="legacy",
        published_at=None,
        deadline_at=None,
        deadline_confidence=DeadlineConfidence.UNKNOWN,
    )
    assert legacy.proof_status == ProofStatus.UNVERIFIED_LEGACY
    assert legacy.verified_at is None
    assert legacy.verification_ref == ""
