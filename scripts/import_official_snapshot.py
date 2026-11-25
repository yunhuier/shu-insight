"""Import a reviewed official snapshot into an explicit local SQLite database."""

from __future__ import annotations

import argparse
from pathlib import Path

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository
from shu_insight.db.schema import Database
from shu_insight.ingest.snapshot import import_snapshot
from shu_insight.sources.registry import load_source_registry

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/official/snapshot-2026-09-24.json")
    parser.add_argument("--database", required=True, type=Path)
    args = parser.parse_args()
    database = Database(args.database.resolve())
    database.initialize()
    policy = OfficialNoticePolicy({
        name: adapter.config for name, adapter in
        load_source_registry(ROOT / "config/sources.toml").items()
    })
    report = import_snapshot(args.manifest, NoticeRepository(database, policy=policy))
    print(f"imported={report.imported_count} unchanged={report.unchanged_count} "
          f"rejected={report.rejected_count}")


if __name__ == "__main__":
    main()
