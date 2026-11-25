"""Import an explicit offline official preload into an explicit local database."""

from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository
from shu_insight.db.schema import SCHEMA_VERSION, Database
from shu_insight.ingest.preload import import_preload
from shu_insight.sources.registry import load_source_registry

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    # Caller owns any active-app process lock and backup. A manifest is checked
    # before creating a destination, so a typo cannot silently initialize one.
    if not args.manifest.is_file():
        parser.error("manifest does not exist")
    registry = load_source_registry(ROOT / "config/sources.toml")
    database = Database(args.database.resolve())
    if database.path.exists():
        try:
            with closing(sqlite3.connect(database.path.as_uri() + "?mode=ro", uri=True)) as conn:
                conn.execute("PRAGMA query_only=ON")
                row = conn.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
                if row is None or row[0] != SCHEMA_VERSION:
                    parser.error("preload refuses schema migration; current schema required")
        except sqlite3.DatabaseError:
            parser.error("destination is not an initialized current-schema database")
    else:
        database.initialize()
    repository = NoticeRepository(database, OfficialNoticePolicy({
        key: adapter.config for key, adapter in registry.items()
    }))
    report = import_preload(args.manifest, repository)
    print(f"imported={report.imported_count} updated={report.updated_count} "
          f"unchanged={report.unchanged_count} rejected={report.rejected_count} "
          f"withdrawn={report.withdrawn_count}")


if __name__ == "__main__":
    main()
