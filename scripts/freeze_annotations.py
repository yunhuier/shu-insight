from __future__ import annotations

import argparse
from pathlib import Path

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository
from shu_insight.db.schema import Database
from shu_insight.evaluation.dataset import freeze_annotations
from shu_insight.sources.registry import load_source_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze human-reviewed SHU Insight annotations.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--database", type=Path,
                        help="verified local notice database; required for formal provenance")
    parser.add_argument("--sources", type=Path, default=Path("config/sources.toml"))
    args = parser.parse_args()
    repository = None
    if args.database is not None:
        database = Database(args.database)
        database.initialize()
        registry = load_source_registry(args.sources)
        policy = OfficialNoticePolicy({key: value.config for key, value in registry.items()})
        repository = NoticeRepository(database, policy=policy)
    dataset = freeze_annotations(args.input, args.output, verified_repository=repository)
    print(f"frozen {len(dataset.rows)} rows; SHA-256 {dataset.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
