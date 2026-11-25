from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository
from shu_insight.db.schema import Database
from shu_insight.sources.registry import load_source_registry


def _rows(repository: NoticeRepository) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for notice in repository.list_verified():
        output.append(
            {
                "id": notice.id,
                "source_id": notice.source_id,
                "title": notice.title,
                "context_excerpt": notice.summary[:500],
                "original_url": notice.canonical_url,
                "predicted_category": notice.categories[0] if notice.categories else "",
                "extracted_entities": [
                    *notice.education_levels,
                    *notice.grades,
                    *notice.colleges,
                    *notice.majors,
                ],
                "published_month": (
                    notice.published_at.strftime("%Y-%m") if notice.published_at else "unknown"
                ),
            }
        )
    return output


def export_annotation_queue(repository: NoticeRepository, output: Path) -> int:
    rows = _rows(repository)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        fieldnames = list(rows[0]) if rows else ["id", "title", "original_url"]
        with output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else value
                        for key, value in row.items()
                    }
                )
    else:
        with output.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a privacy-minimized annotation queue.")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=Path("config/sources.toml"))
    args = parser.parse_args()
    database = Database(args.database)
    database.initialize()
    registry = load_source_registry(args.sources)
    policy = OfficialNoticePolicy({key: item.config for key, item in registry.items()})
    count = export_annotation_queue(NoticeRepository(database, policy=policy), args.output)
    print(f"wrote {count} currently verified official notice rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
