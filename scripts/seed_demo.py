from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from shu_insight.compliance.detector import scan_sensitive_text
from shu_insight.db.repositories import NoticeRepository, PersonaRepository, SourceRepository
from shu_insight.db.schema import Database
from shu_insight.domain.models import DeadlineConfidence, NoticeDraft
from shu_insight.profiles.service import DEMO_PERSONAS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "demo" / "notices.json"
DEFAULT_DATABASE = PROJECT_ROOT / "data" / "shu_insight.sqlite3"
ALLOWED_SOURCE_IDS = frozenset({"shu_main", "undergraduate", "graduate", "computer"})


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("demo data must be a JSON list")
    records = [dict(item) for item in payload if isinstance(item, dict)]
    if len(records) != len(payload) or not 12 <= len(records) <= 20:
        raise ValueError("demo data must contain 12 to 20 object records")
    return records


def _reset(database: Database) -> None:
    tables = (
        "recommendation_items",
        "recommendation_runs",
        "annotations",
        "duplicate_cluster_members",
        "duplicate_clusters",
        "notice_versions",
        "notices",
        "source_status",
        "sources",
        "personas",
    )
    with database.transaction() as connection:
        for table in tables:
            connection.execute(f"DELETE FROM {table}")  # noqa: S608


def _offset_date(now: datetime, value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("date offsets must be integers or null")
    return now + timedelta(days=value)


def _validate_record(record: dict[str, Any]) -> None:
    if record.get("provenance") != "synthetic_demo":
        raise ValueError("every demo record must declare synthetic_demo provenance")
    if record.get("source_id") not in ALLOWED_SOURCE_IDS:
        raise ValueError("demo record uses an unknown source id")
    if urlsplit(str(record.get("canonical_url", ""))).hostname != "demo.invalid":
        raise ValueError("demo URLs must use the reserved demo.invalid host")
    if not str(record.get("title", "")).startswith("[演示]"):
        raise ValueError("demo titles must be visibly marked")
    text = f"{record.get('title', '')}\n{record.get('normalized_text', '')}"
    scan = scan_sensitive_text(text)
    if scan.blocked:
        raise ValueError(f"demo record contains blocked content: {', '.join(scan.kinds)}")


def seed_demo(
    database_path: Path,
    *,
    reset: bool = False,
    data_path: Path = DEFAULT_DATA,
    now: datetime | None = None,
) -> Database:
    database = Database(database_path.resolve())
    database.initialize()
    if reset:
        _reset(database)

    reference_time = (now or datetime.now(UTC)).replace(microsecond=0)
    notice_repository = NoticeRepository(database)
    for record in _load_records(data_path):
        _validate_record(record)
        tags = tuple(dict.fromkeys((*record.get("topic_tags", ()), "synthetic_demo")))
        notice_repository.upsert(
            NoticeDraft(
                source_id=str(record["source_id"]),
                canonical_url=str(record["canonical_url"]),
                title=str(record["title"]),
                normalized_text=str(record["normalized_text"]),
                published_at=_offset_date(reference_time, record.get("published_offset_days")),
                deadline_at=_offset_date(reference_time, record.get("deadline_offset_days")),
                deadline_confidence=DeadlineConfidence(record["deadline_confidence"]),
                location=str(record.get("location", "")),
                department=str(record.get("department", "")),
                column_name="离线演示",
                education_levels=tuple(record.get("education_levels", ())),
                grades=tuple(record.get("grades", ())),
                colleges=tuple(record.get("colleges", ())),
                majors=tuple(record.get("majors", ())),
                categories=tuple(record.get("categories", ())),
                topic_tags=tags,
            ),
            refresh_mining=False,
        )
    notice_repository.refresh_mining()

    persona_repository = PersonaRepository(database)
    for persona in DEMO_PERSONAS:
        persona_repository.save(persona)

    source_repository = SourceRepository(database)
    for source_id in sorted(ALLOWED_SOURCE_IDS):
        source_repository.record_status(
            {
                "source_id": source_id,
                "state": "synthetic_demo",
                "checked_at": reference_time,
                "message": "synthetic_demo：离线种子，无网络请求",
            }
        )
    return database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed safe, fictional offline demo data.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="clear local application tables before inserting the fictional demo",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    database = seed_demo(args.database, reset=args.reset, data_path=args.data)
    count = NoticeRepository(database).count()
    print(f"seeded {count} fictional notices into {database.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
