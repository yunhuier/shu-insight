from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 7

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
  version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  base_url TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS update_runs (
  id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  state TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_candidates (
  source_id TEXT NOT NULL REFERENCES sources(id),
  canonical_url TEXT NOT NULL,
  first_seen_at TEXT NOT NULL,
  last_attempt_at TEXT,
  transient_failures INTEGER NOT NULL DEFAULT 0,
  paused_until TEXT,
  PRIMARY KEY(source_id, canonical_url)
);

CREATE TABLE IF NOT EXISTS source_status (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  state TEXT NOT NULL,
  checked_at TEXT NOT NULL,
  message TEXT NOT NULL DEFAULT '',
  saved_count INTEGER NOT NULL DEFAULT 0,
  updated_count INTEGER NOT NULL DEFAULT 0,
  skipped_count INTEGER NOT NULL DEFAULT 0,
  skip_reasons_json TEXT NOT NULL DEFAULT '{}',
  run_id TEXT REFERENCES update_runs(id),
  list_pages_read INTEGER NOT NULL DEFAULT 0,
  request_count INTEGER NOT NULL DEFAULT 0,
  queued_count INTEGER NOT NULL DEFAULT 0,
  coverage_code TEXT NOT NULL DEFAULT '',
  request_breakdown_json TEXT NOT NULL DEFAULT '{}',
  discovered_count INTEGER NOT NULL DEFAULT 0,
  parsed_count INTEGER NOT NULL DEFAULT 0,
  listing_count INTEGER NOT NULL DEFAULT 0,
  pending_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS notices (
  id TEXT PRIMARY KEY,
  canonical_url TEXT NOT NULL UNIQUE,
  source_id TEXT NOT NULL REFERENCES sources(id),
  title TEXT NOT NULL,
  published_at TEXT,
  deadline_at TEXT,
  deadline_confidence TEXT NOT NULL,
  summary TEXT NOT NULL,
  location TEXT NOT NULL DEFAULT '',
  department TEXT NOT NULL DEFAULT '',
  column_name TEXT NOT NULL DEFAULT '',
  education_levels_json TEXT NOT NULL DEFAULT '[]',
  grades_json TEXT NOT NULL DEFAULT '[]',
  colleges_json TEXT NOT NULL DEFAULT '[]',
  majors_json TEXT NOT NULL DEFAULT '[]',
  categories_json TEXT NOT NULL DEFAULT '[]',
  topic_tags_json TEXT NOT NULL DEFAULT '[]',
  content_fingerprint TEXT NOT NULL,
  summary_simhash INTEGER NOT NULL,
  parse_state TEXT NOT NULL,
  fetched_at TEXT NOT NULL,
  proof_status TEXT NOT NULL DEFAULT 'unverified_legacy',
  verified_at TEXT,
  verification_ref TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS notice_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  notice_id TEXT NOT NULL REFERENCES notices(id) ON DELETE CASCADE,
  content_fingerprint TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS duplicate_relations (
  left_id TEXT NOT NULL REFERENCES notices(id) ON DELETE CASCADE,
  right_id TEXT NOT NULL REFERENCES notices(id) ON DELETE CASCADE,
  left_fingerprint TEXT NOT NULL,
  right_fingerprint TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(left_id, right_id, left_fingerprint, right_fingerprint)
);

CREATE TABLE IF NOT EXISTS duplicate_clusters (
  id TEXT PRIMARY KEY,
  confidence REAL NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS duplicate_cluster_members (
  cluster_id TEXT NOT NULL REFERENCES duplicate_clusters(id) ON DELETE CASCADE,
  notice_id TEXT NOT NULL REFERENCES notices(id) ON DELETE CASCADE,
  relation TEXT NOT NULL,
  PRIMARY KEY (cluster_id, notice_id)
);

CREATE TABLE IF NOT EXISTS personas (
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  notice_id TEXT NOT NULL REFERENCES notices(id) ON DELETE CASCADE,
  payload_json TEXT NOT NULL,
  reviewed_by TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  tags_json TEXT NOT NULL DEFAULT '[]',
  keywords_json TEXT NOT NULL DEFAULT '[]',
  terms_json TEXT NOT NULL DEFAULT '[]',
  persona_id TEXT,
  result_count INTEGER NOT NULL DEFAULT 0,
  top_categories_json TEXT NOT NULL DEFAULT '[]',
  top_tags_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS article_reads (
  notice_id TEXT PRIMARY KEY REFERENCES notices(id) ON DELETE CASCADE,
  first_read_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendation_runs (
  id TEXT PRIMARY KEY,
  persona_id TEXT REFERENCES personas(id) ON DELETE SET NULL,
  parameters_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendation_items (
  run_id TEXT NOT NULL REFERENCES recommendation_runs(id) ON DELETE CASCADE,
  notice_id TEXT NOT NULL REFERENCES notices(id) ON DELETE CASCADE,
  rank INTEGER NOT NULL,
  score REAL NOT NULL,
  breakdown_json TEXT NOT NULL,
  PRIMARY KEY (run_id, notice_id)
);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name NOT LIKE 'sqlite_%'"
                )
            }
            if not tables:
                connection.executescript(SCHEMA_SQL)
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS source_status_run_id_idx ON source_status(run_id)"
                )
                connection.execute("INSERT INTO schema_meta(version) VALUES (?)", (SCHEMA_VERSION,))
                return
            if "schema_meta" not in tables:
                raise RuntimeError("Existing database has no schema_meta table")
            row = connection.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
            if row is None:
                raise RuntimeError("Existing database has no schema version")
            version = row["version"]
            if version == SCHEMA_VERSION:
                return
            if version not in (1, 2, 3, 4, 5, 6):
                raise RuntimeError(
                    f"Unsupported database schema {version}; expected {SCHEMA_VERSION}"
                )

            self._backup_before_migration(version)
            connection.execute("BEGIN IMMEDIATE")
            try:
                if version == 1:
                    # Older v1 databases may be missing tables added later.
                    self._create_missing_tables(connection)
                    self._migrate_v1_versions(connection)
                if version <= 2:
                    self._migrate_v2_to_v3(connection)
                if version <= 3:
                    self._migrate_v3_to_v4(connection)
                if version <= 4:
                    self._migrate_v4_to_v5(connection)
                if version <= 5:
                    self._migrate_v5_to_v6(connection)
                self._migrate_v6_to_v7(connection)
                connection.execute("UPDATE schema_meta SET version = ?", (SCHEMA_VERSION,))
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    @staticmethod
    def _create_missing_tables(connection: sqlite3.Connection) -> None:
        statement = ""
        for line in SCHEMA_SQL.splitlines(keepends=True):
            statement += line
            if sqlite3.complete_statement(statement):
                connection.execute(statement)
                statement = ""

    def _backup_before_migration(self, version: int) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        backup_path = (
            self.path.parent / "backups" / f"{self.path.stem}-v{version}-{timestamp}.sqlite3"
        )
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as source, closing(
            sqlite3.connect(backup_path)
        ) as target:
            source.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("database backup failed integrity check")
        return backup_path

    @staticmethod
    def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(notices)")
        }
        for name, definition in (
            ("proof_status", "TEXT NOT NULL DEFAULT 'unverified_legacy'"),
            ("verified_at", "TEXT"),
            ("verification_ref", "TEXT NOT NULL DEFAULT ''"),
        ):
            if name not in columns:
                connection.execute(f"ALTER TABLE notices ADD COLUMN {name} {definition}")

        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(source_status)")
        }
        for name, definition in (
            ("saved_count", "INTEGER NOT NULL DEFAULT 0"),
            ("updated_count", "INTEGER NOT NULL DEFAULT 0"),
            ("skipped_count", "INTEGER NOT NULL DEFAULT 0"),
            ("skip_reasons_json", "TEXT NOT NULL DEFAULT '{}'"),
        ):
            if name not in columns:
                connection.execute(f"ALTER TABLE source_status ADD COLUMN {name} {definition}")

    @staticmethod
    def _migrate_v3_to_v4(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS update_runs (
              id TEXT PRIMARY KEY,
              started_at TEXT NOT NULL,
              finished_at TEXT,
              state TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_candidates (
              source_id TEXT NOT NULL REFERENCES sources(id),
              canonical_url TEXT NOT NULL,
              first_seen_at TEXT NOT NULL,
              last_attempt_at TEXT,
              transient_failures INTEGER NOT NULL DEFAULT 0,
              paused_until TEXT,
              PRIMARY KEY(source_id, canonical_url)
            )
            """
        )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(source_status)")
        }
        for name, definition in (
            ("run_id", "TEXT REFERENCES update_runs(id)"),
            ("list_pages_read", "INTEGER NOT NULL DEFAULT 0"),
            ("request_count", "INTEGER NOT NULL DEFAULT 0"),
            ("queued_count", "INTEGER NOT NULL DEFAULT 0"),
            ("coverage_code", "TEXT NOT NULL DEFAULT ''"),
        ):
            if name not in columns:
                connection.execute(f"ALTER TABLE source_status ADD COLUMN {name} {definition}")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS source_status_run_id_idx ON source_status(run_id)"
        )

    @staticmethod
    def _migrate_v4_to_v5(connection: sqlite3.Connection) -> None:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(source_status)")}
        for name, definition in (
            ("request_breakdown_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("discovered_count", "INTEGER NOT NULL DEFAULT 0"),
            ("parsed_count", "INTEGER NOT NULL DEFAULT 0"),
            ("listing_count", "INTEGER NOT NULL DEFAULT 0"),
            ("pending_count", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if name not in columns:
                connection.execute(f"ALTER TABLE source_status ADD COLUMN {name} {definition}")

    @staticmethod
    def _migrate_v5_to_v6(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS search_history (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL,
              tags_json TEXT NOT NULL DEFAULT '[]',
              keywords_json TEXT NOT NULL DEFAULT '[]',
              terms_json TEXT NOT NULL DEFAULT '[]',
              persona_id TEXT,
              result_count INTEGER NOT NULL DEFAULT 0,
              top_categories_json TEXT NOT NULL DEFAULT '[]',
              top_tags_json TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS search_history_created_at_idx "
            "ON search_history(created_at)"
        )

    @staticmethod
    def _migrate_v6_to_v7(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS article_reads ("
            "notice_id TEXT PRIMARY KEY REFERENCES notices(id) ON DELETE CASCADE, "
            "first_read_at TEXT NOT NULL)"
        )

    @staticmethod
    def _migrate_v1_versions(connection: sqlite3.Connection) -> None:
        # A previous interrupted migration may already have renamed the v1
        # table; resume from that state within the initializer's transaction.
        renamed = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'notice_versions_v1'"
        ).fetchone()
        if renamed is None:
            connection.execute(
                "ALTER TABLE notice_versions RENAME TO notice_versions_v1"
            )
            connection.execute(
                """
                CREATE TABLE notice_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    notice_id TEXT NOT NULL REFERENCES notices(id) ON DELETE CASCADE,
                    content_fingerprint TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
        connection.execute(
            "INSERT OR IGNORE INTO notice_versions "
            "(id, notice_id, content_fingerprint, payload_json, created_at) "
            "SELECT id, notice_id, content_fingerprint, payload_json, created_at "
            "FROM notice_versions_v1"
        )
        connection.execute("DROP TABLE notice_versions_v1")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
