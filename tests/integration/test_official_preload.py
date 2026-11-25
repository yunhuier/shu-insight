"""Validate the actual reusable stock, independent of any network or source DB."""

import json
import socket
from collections import Counter
from pathlib import Path

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository
from shu_insight.db.schema import Database
from shu_insight.ingest.preload import import_preload
from shu_insight.sources.registry import load_source_registry

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "data/official/preload-2026-10-25.json"


def test_real_pack_imports_all_four_sources_offline_without_duplicate_versions(
    tmp_path, monkeypatch,
):
    assert PACK.is_file(), "real reviewed preload has not been generated"

    def forbidden(*args, **kwargs):
        raise AssertionError("preload attempted network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    registry = load_source_registry(ROOT / "config/sources.toml")
    database = Database(tmp_path / "preload.sqlite3")
    database.initialize()
    repository = NoticeRepository(database, OfficialNoticePolicy({
        key: adapter.config for key, adapter in registry.items()
    }))
    first = import_preload(PACK, repository)
    assert (first.imported_count, first.updated_count, first.rejected_count) == (74, 0, 0)
    records = repository.list_browseable()
    assert len({record.canonical_url for record in records}) == 74
    assert Counter(record.source_id for record in records) == {
        "shu_main": 25, "undergraduate": 42, "graduate": 5, "computer": 2,
    }
    assert Counter(record.proof_status for record in records) == {
        "verified_live": 46, "verified_snapshot": 7, "verified_listing": 21,
    }
    assert all(not record.summary and record.deadline_at is None
               for record in records if record.parse_state == "listing_only")
    second = import_preload(PACK, repository)
    assert (second.imported_count, second.updated_count, second.unchanged_count,
            second.rejected_count) == (0, 0, 74, 0)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM notice_versions").fetchone()[0] == 74
    payload = json.loads(PACK.read_text(encoding="utf-8"))
    saved = [entry for entry in payload["entries"]
             if entry["provenance"]["kind"] == "offline_article_review"]
    assert len(saved) == 1
    assert saved[0]["fetched_at"] is None
    assert saved[0]["published_at"] == "2026-09-04T00:00:00+08:00"
    assert saved[0]["provenance"]["sha256"] == (
        "8ce4b15bcd70292dfc7e642f40e90475f7516fe2e2d0e2d26f6a90f94befe549"
    )
    assert repository.find_by_url('https://bksy.shu.edu.cn/info/1015/322264.htm') is None
    assert payload['attachment_exclusions'][0]['canonical_url'] == (
        'https://bksy.shu.edu.cn/info/1015/322264.htm'
    )
