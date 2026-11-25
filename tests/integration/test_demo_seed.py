import json
from pathlib import Path

from scripts.export_annotation_queue import _rows
from scripts.seed_demo import seed_demo
from shu_insight.compliance.detector import scan_sensitive_text
from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository, PersonaRepository
from shu_insight.services.notices import NoticeQuery, NoticeService
from shu_insight.sources.registry import load_source_registry

DEMO_JSON = Path("data/demo/notices.json")


def test_demo_seed_contains_only_bounded_synthetic_data(tmp_path: Path) -> None:
    database = seed_demo(tmp_path / "demo.sqlite3", reset=True)
    notices = NoticeRepository(database).list_all()
    raw_records = json.loads(DEMO_JSON.read_text(encoding="utf-8"))

    assert 12 <= len(notices) <= 20
    assert len(raw_records) == len(notices)
    assert all(record["provenance"] == "synthetic_demo" for record in raw_records)
    assert all("synthetic_demo" in notice.topic_tags for notice in notices)
    assert all(
        not scan_sensitive_text(f"{notice.title}\n{notice.summary}").blocked
        for notice in notices
    )
    from shu_insight.profiles.service import DEMO_PERSONAS

    assert len(PersonaRepository(database).list_all()) == len(DEMO_PERSONAS)


def test_demo_seed_reset_is_repeatable(tmp_path: Path) -> None:
    database_path = tmp_path / "demo.sqlite3"

    seed_demo(database_path, reset=True)
    database = seed_demo(database_path, reset=True)

    assert NoticeRepository(database).count() == 16
def test_demo_seed_does_not_record_live_source_success(tmp_path):
    from shu_insight.db.repositories import SourceRepository

    database = seed_demo(tmp_path / "demo-status.sqlite3", reset=True)
    states = SourceRepository(database).latest_statuses()
    assert len(states) == 4
    assert {item["state"] for item in states} == {"not_checked"}
    assert all(item["checked_at"] is None for item in states)
    with database.connect() as connection:
        stored_states = {row[0] for row in connection.execute(
            "SELECT DISTINCT state FROM source_status"
        )}
    assert stored_states == {"synthetic_demo"}


def test_demo_seed_is_absent_from_service_and_annotation_export(tmp_path: Path) -> None:
    database = seed_demo(tmp_path / "demo-visibility.sqlite3", reset=True)
    registry = load_source_registry(Path("config/sources.toml"))
    policy = OfficialNoticePolicy({key: item.config for key, item in registry.items()})
    repository = NoticeRepository(database, policy=policy)
    service = NoticeService(repository, authority_by_source_id={})
    assert service.all() == []
    assert service.latest() == []
    assert service.upcoming().all_notices == []
    assert service.recommend(NoticeQuery(terms=("通知",))) == []
    assert _rows(repository) == []
