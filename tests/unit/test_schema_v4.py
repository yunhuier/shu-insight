from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository
from shu_insight.db.schema import Database
from shu_insight.domain.models import DiscoveredNotice, ParseState, ProofStatus
from shu_insight.sources.registry import load_source_registry


def test_v4_database_round_trips_listing_state(tmp_path):
    path = tmp_path / "listing-v4.sqlite3"
    database = Database(path)
    database.initialize()
    policy = OfficialNoticePolicy({
        key: item.config for key, item in load_source_registry(Path("config/sources.toml")).items()
    })
    NoticeRepository(database, policy=policy).upsert_listing(
        DiscoveredNotice("graduate", "https://gs.shu.edu.cn/info/1029/173242.htm", "课程免修通知"),
        list_url="https://gs.shu.edu.cn/xwlb/sy.htm",
        checked_at=datetime(2026, 10, 19, tzinfo=UTC),
    )
    reopened = Database(path)
    reopened.initialize()
    record, = NoticeRepository(reopened, policy=policy).list_browseable()
    assert record.parse_state == ParseState.LISTING_ONLY
    assert record.proof_status == ProofStatus.VERIFIED_LISTING
    with reopened.connect() as connection:
        assert connection.execute("SELECT version FROM schema_meta").fetchone()[0] == 5
    assert not (tmp_path / "backups").exists()


def make_legacy_database(tmp_path: Path, version: int) -> Path:
    path = tmp_path / f"v{version}.sqlite3"
    schema = Path("tests/fixtures/db/schema_v2.sql").read_text(encoding="utf-8")
    if version == 1:
        schema = schema.replace(
            "created_at TEXT NOT NULL\n);\n\nCREATE TABLE IF NOT EXISTS duplicate_relations",
            "created_at TEXT NOT NULL,\n  UNIQUE(notice_id, content_fingerprint)\n);\n\n"
            "CREATE TABLE IF NOT EXISTS duplicate_relations",
        )
    with sqlite3.connect(path) as connection:
        connection.executescript(schema)
        connection.execute("INSERT INTO schema_meta VALUES (?)", (version,))
        connection.execute(
            "INSERT INTO sources VALUES ('computer', 'Computer', "
            "'https://cs.shu.edu.cn/', '2026-10-05')"
        )
        connection.execute(
            "INSERT INTO notices(id, canonical_url, source_id, title, deadline_confidence, "
            "summary, content_fingerprint, summary_simhash, parse_state, fetched_at) "
            "VALUES ('notice-1', 'https://example.invalid/notice', 'computer', 'Kept title', "
            "'unknown', 'Kept summary', 'hash-1', 123, 'parsed', '2026-10-05')"
        )
        connection.execute(
            "INSERT INTO notice_versions VALUES "
            "(7, 'notice-1', 'hash-1', '{\"title\":\"Kept title\"}', '2026-10-05')"
        )
        connection.execute(
            "INSERT INTO personas VALUES "
            "('kept-persona', 'Kept persona', '{\"grade\":2024}', '2026-10-05')"
        )
        connection.execute(
            "INSERT INTO source_status(id, source_id, state, checked_at, message) "
            "VALUES (9, 'computer', 'success', '2026-10-05', 'Historical check')"
        )
        if version == 3:
            connection.executescript(
                """
                ALTER TABLE notices ADD COLUMN proof_status
                  TEXT NOT NULL DEFAULT 'unverified_legacy';
                ALTER TABLE notices ADD COLUMN verified_at TEXT;
                ALTER TABLE notices ADD COLUMN verification_ref TEXT NOT NULL DEFAULT '';
                ALTER TABLE source_status ADD COLUMN saved_count INTEGER NOT NULL DEFAULT 0;
                ALTER TABLE source_status ADD COLUMN updated_count INTEGER NOT NULL DEFAULT 0;
                ALTER TABLE source_status ADD COLUMN skipped_count INTEGER NOT NULL DEFAULT 0;
                ALTER TABLE source_status ADD COLUMN skip_reasons_json TEXT NOT NULL DEFAULT '{}';
                UPDATE notices SET proof_status = 'verified_snapshot',
                  verified_at = '2026-10-05T00:00:00Z', verification_ref = 'archive:sha256:kept';
                UPDATE source_status SET saved_count = 3, updated_count = 2, skipped_count = 1,
                  skip_reasons_json = '{"already_seen":1}';
                """
            )
    return path


def database_dump(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        return "\n".join(connection.iterdump())


def test_copied_v4_fixture_migrates_to_v5_and_backup_restores_all_original_state(tmp_path):
    original = make_legacy_database(tmp_path, 3)
    with sqlite3.connect(original) as connection:
        Database._migrate_v3_to_v4(connection)
        connection.execute("UPDATE schema_meta SET version = 4")
        connection.execute("INSERT INTO pending_candidates VALUES "
                           "('computer','https://cs.shu.edu.cn/info/1/2.htm',"
                           "'2026-10-05',NULL,2,'2026-10-12')")
        for ident, proof, state in (("listing", "verified_listing", "listing_only"),
                                    ("tombstone", "attachment_excluded", "listing_only")):
            connection.execute(
                "INSERT INTO notices(id,canonical_url,source_id,title,deadline_confidence,"
                "summary,content_fingerprint,summary_simhash,parse_state,fetched_at,proof_status) "
                "VALUES (?,?,'computer','public title','unknown','','fixture',0,?,'2026-10-05',?)",
                (ident, f"https://cs.shu.edu.cn/info/1/{ident}.htm", state, proof),
            )
    before = database_dump(original)
    copied = tmp_path / "copied-v4.sqlite3"
    with closing(sqlite3.connect(original)) as source, closing(sqlite3.connect(copied)) as target:
        source.backup(target)
    Database(copied).initialize()
    assert database_dump(original) == before
    backups = list((tmp_path / "backups").glob("copied-v4-v4-*.sqlite3"))
    assert len(backups) == 1 and database_dump(backups[0]) == before
    restored = tmp_path / "restored-v4.sqlite3"
    with (closing(sqlite3.connect(backups[0])) as source,
          closing(sqlite3.connect(restored)) as target):
        source.backup(target)
        assert target.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert database_dump(restored) == before
    with sqlite3.connect(copied) as connection, sqlite3.connect(original) as old:
        assert connection.execute("SELECT version FROM schema_meta").fetchone()[0] == 5
        for table in (
            "notices", "notice_versions", "personas", "pending_candidates", "update_runs",
        ):
            assert connection.execute(f"SELECT * FROM {table}").fetchall() == old.execute(
                f"SELECT * FROM {table}"
            ).fetchall()
        assert connection.execute("SELECT request_breakdown_json,discovered_count,parsed_count,"
                                  "listing_count,pending_count FROM source_status").fetchone() == (
                                      '{}', 0, 0, 0, 0,
                                  )
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def assert_v4_records(connection: sqlite3.Connection) -> None:
    assert connection.execute("SELECT version FROM schema_meta").fetchone()[0] == 5
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert connection.execute(
        "SELECT run_id, list_pages_read, request_count, queued_count, coverage_code "
        "FROM source_status WHERE id = 9"
    ).fetchone() == (None, 0, 0, 0, "")
    assert connection.execute("SELECT * FROM update_runs").fetchall() == []
    assert connection.execute("SELECT * FROM pending_candidates").fetchall() == []
    assert connection.execute(
        "SELECT id, canonical_url, source_id, title, summary, content_fingerprint, "
        "summary_simhash, parse_state, fetched_at FROM notices"
    ).fetchone() == (
        "notice-1", "https://example.invalid/notice", "computer", "Kept title", "Kept summary",
        "hash-1", 123, "parsed", "2026-10-05",
    )
    assert connection.execute(
        "SELECT payload_json FROM personas WHERE id = 'kept-persona'"
    ).fetchone()[0] == '{"grade":2024}'
    assert connection.execute(
        "SELECT id, notice_id, content_fingerprint, payload_json, created_at FROM notice_versions"
    ).fetchone() == (7, "notice-1", "hash-1", '{"title":"Kept title"}', "2026-10-05")
    assert connection.execute(
        "SELECT state, checked_at, message FROM source_status WHERE id = 9"
    ).fetchone() == ("success", "2026-10-05", "Historical check")


@pytest.mark.parametrize("version", [1, 2, 3])
def test_legacy_upgrade_preserves_data_and_backs_up_before_any_ddl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: int,
) -> None:
    path = make_legacy_database(tmp_path, version)
    original_dump = database_dump(path)
    original_connect = Database.connect
    original_backup = Database._backup_before_migration
    backup_complete = False
    ddl_backups: list[bool] = []

    def traced_connect(database: Database) -> sqlite3.Connection:
        connection = original_connect(database)

        def record_ddl(statement: str) -> None:
            if statement.lstrip().split()[0].upper() in {"CREATE", "ALTER", "DROP"}:
                ddl_backups.append(backup_complete)

        connection.set_trace_callback(record_ddl)
        return connection

    def checked_backup(database: Database, old_version: int) -> Path:
        nonlocal backup_complete
        backup_path = original_backup(database, old_version)
        assert database_dump(backup_path) == original_dump
        with sqlite3.connect(backup_path) as backup:
            assert backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        backup_complete = True
        return backup_path

    monkeypatch.setattr(Database, "connect", traced_connect)
    monkeypatch.setattr(Database, "_backup_before_migration", checked_backup)

    Database(path).initialize()
    Database(path).initialize()

    assert ddl_backups and all(ddl_backups)
    backups = list((tmp_path / "backups").glob("*.sqlite3"))
    assert len(backups) == 1
    assert database_dump(backups[0]) == original_dump
    with sqlite3.connect(path) as connection:
        assert_v4_records(connection)
        proof = connection.execute(
            "SELECT proof_status, verified_at, verification_ref FROM notices"
        ).fetchone()
        expected_proof = (
            ("verified_snapshot", "2026-10-05T00:00:00Z", "archive:sha256:kept")
            if version == 3 else ("unverified_legacy", None, "")
        )
        assert proof == expected_proof
        counts = connection.execute(
            "SELECT saved_count, updated_count, skipped_count, skip_reasons_json FROM source_status"
        ).fetchone()
        assert counts == ((3, 2, 1, '{"already_seen":1}') if version == 3 else (0, 0, 0, "{}"))
        # v1's historical fingerprint uniqueness must still be removed by its upgrade.
        connection.execute(
            "INSERT INTO notice_versions(notice_id, content_fingerprint, payload_json, created_at) "
            "VALUES ('notice-1', 'hash-1', '{}', '2026-10-06')"
        )


@pytest.mark.parametrize("version", [1, 2, 3])
def test_backup_failure_preserves_entire_original_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: int,
) -> None:
    path = make_legacy_database(tmp_path, version)
    original_dump = database_dump(path)

    def failing_backup(database: Database, old_version: int) -> Path:
        raise OSError("injected backup failure")

    monkeypatch.setattr(Database, "_backup_before_migration", failing_backup)
    with pytest.raises(OSError, match="injected backup failure"):
        Database(path).initialize()

    assert database_dump(path) == original_dump


@pytest.mark.parametrize("version", [1, 2, 3])
def test_v4_interruption_rolls_back_entire_upgrade_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: int,
) -> None:
    path = make_legacy_database(tmp_path, version)
    original_dump = database_dump(path)
    migrate_v4 = getattr(Database, "_migrate_v3_to_v4", None)

    def interrupted_migration(connection: sqlite3.Connection) -> None:
        if migrate_v4 is not None:
            migrate_v4(connection)
        assert connection.execute("SELECT version FROM schema_meta").fetchone()[0] == version
        connection.execute("CREATE TABLE migration_probe (id TEXT)")
        connection.execute("UPDATE source_status SET message = 'must roll back'")
        raise RuntimeError("injected v4 interruption")

    with monkeypatch.context() as patch:
        patch.setattr(
            Database, "_migrate_v3_to_v4", staticmethod(interrupted_migration), raising=False,
        )
        with pytest.raises(RuntimeError, match="injected v4 interruption"):
            Database(path).initialize()

    assert database_dump(path) == original_dump
    backups = list((tmp_path / "backups").glob("*.sqlite3"))
    assert len(backups) == 1
    assert database_dump(backups[0]) == original_dump
    Database(path).initialize()
    with sqlite3.connect(path) as connection:
        assert_v4_records(connection)


def test_new_database_can_store_runs_queue_and_linked_status(tmp_path: Path) -> None:
    path = tmp_path / "new.sqlite3"
    database = Database(path)
    database.initialize()

    with database.connect() as connection:
        assert connection.execute("SELECT version FROM schema_meta").fetchone()[0] == 5
        connection.execute(
            "INSERT INTO sources VALUES ('computer', 'Computer', "
            "'https://cs.shu.edu.cn/', '2026-10-05')"
        )
        connection.execute(
            "INSERT INTO update_runs(id, started_at, state) "
            "VALUES ('run-1', '2026-10-06T00:00:00Z', 'running')"
        )
        assert tuple(connection.execute("SELECT * FROM update_runs").fetchone()) == (
            "run-1", "2026-10-06T00:00:00Z", None, "running",
        )
        connection.execute(
            "INSERT INTO pending_candidates(source_id, canonical_url, first_seen_at) "
            "VALUES ('computer', 'https://example.invalid/new', '2026-10-06')"
        )
        assert tuple(connection.execute("SELECT * FROM pending_candidates").fetchone()) == (
            "computer", "https://example.invalid/new", "2026-10-06", None, 0, None,
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO pending_candidates(source_id, canonical_url, first_seen_at) "
                "VALUES ('computer', 'https://example.invalid/new', '2026-10-07')"
            )
        connection.execute(
            "INSERT INTO source_status(source_id, state, checked_at, run_id, list_pages_read, "
            "request_count, queued_count, coverage_code) "
            "VALUES ('computer', 'success', '2026-10-06', 'run-1', 2, 4, 1, 'complete')"
        )
        row = connection.execute(
            "SELECT run_id, list_pages_read, request_count, queued_count, coverage_code "
            "FROM source_status"
        ).fetchone()
        assert tuple(row) == ("run-1", 2, 4, 1, "complete")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE source_status SET run_id = 'missing-run'")
        run_indexes = [
            index[1] for index in connection.execute("PRAGMA index_list(source_status)")
            if [column[2] for column in connection.execute(f"PRAGMA index_info({index[1]})")]
            == ["run_id"]
        ]
        assert run_indexes
        connection.execute(
            "UPDATE pending_candidates SET last_attempt_at = '2026-10-07', "
            "transient_failures = 2, paused_until = '2026-10-08'"
        )
        connection.execute(
            "UPDATE update_runs SET state = 'finished', finished_at = '2026-10-07'"
        )
    database.initialize()
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT last_attempt_at, transient_failures, paused_until FROM pending_candidates"
        ).fetchone() == ("2026-10-07", 2, "2026-10-08")
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not (tmp_path / "backups").exists()


def test_partial_v4_tables_and_columns_are_preserved_while_missing_ones_are_added(
    tmp_path: Path,
) -> None:
    path = make_legacy_database(tmp_path, 3)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE update_runs (
              id TEXT PRIMARY KEY,
              started_at TEXT NOT NULL,
              finished_at TEXT,
              state TEXT NOT NULL
            );
            INSERT INTO update_runs VALUES ('kept-run', '2026-10-05', NULL, 'running');
            ALTER TABLE source_status ADD COLUMN run_id TEXT REFERENCES update_runs(id);
            ALTER TABLE source_status ADD COLUMN list_pages_read INTEGER NOT NULL DEFAULT 0;
            UPDATE source_status SET list_pages_read = 2;
            """
        )
    Database(path).initialize()
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT version FROM schema_meta").fetchone()[0] == 5
        assert connection.execute("SELECT * FROM update_runs").fetchone() == (
            "kept-run", "2026-10-05", None, "running",
        )
        assert connection.execute(
            "SELECT run_id, list_pages_read, request_count, queued_count, coverage_code "
            "FROM source_status"
        ).fetchone() == (None, 2, 0, 0, "")
        assert connection.execute("SELECT COUNT(*) FROM pending_candidates").fetchone()[0] == 0
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_sparse_v1_creates_missing_v4_tables_without_duplicate_ddl(tmp_path: Path) -> None:
    path = tmp_path / "sparse-v1.sqlite3"
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
    Database(path).initialize()
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT version FROM schema_meta").fetchone()[0] == 5
        assert connection.execute("SELECT COUNT(*) FROM pending_candidates").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM update_runs").fetchone()[0] == 0
        assert "run_id" in {
            column[1] for column in connection.execute("PRAGMA table_info(source_status)")
        }
