from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository
from shu_insight.db.schema import Database
from shu_insight.domain.models import DeadlineConfidence, NoticeDraft, ProofStatus, Verification
from shu_insight.ingest.snapshot import import_snapshot
from shu_insight.sources.registry import load_source_registry

CONFIG = Path(__file__).resolve().parents[2] / "config" / "sources.toml"
SNAPSHOT = Path(__file__).resolve().parents[2] / "data" / "official" / "snapshot-2026-09-24.json"
CHECKS = {name: True for name in (
    "url", "title", "date", "audience", "action", "summary", "sensitivity"
)}
REAL_URL = "https://bksy.shu.edu.cn/info/1015/321714.htm"
REAL_TITLE = "上海大学2026级本科新生秋季学期选课通知"
OTHER_REAL_URL = "https://bksy.shu.edu.cn/info/1015/320424.htm"
OTHER_REAL_TITLE = "2026-2027学年秋季学期本科生选课通知"
REAL_SUMMARY = (
    "该通知面向上海大学2026级本科新生及各教学单位，说明2026至2027学年秋季学期本科新生网上选课安排。"
    "第一轮选课时间为2026年9月3日9时至17时30分，第二轮为9月6日13时至17时30分；截至核对日，两轮选课均已结束。"
    "通知要求学生使用学校统一身份认证账号登录选课系统，并按培养方案、课程代码、校区和先修要求核对所选课程。"
    "文中还说明35学分上限、秋季学期前两周仅可退一门课的自由退课规则，以及正式上课前确认课表和教学地点的要求。"
)


def _repository(tmp_path: Path) -> NoticeRepository:
    database = Database(tmp_path / "snapshot.sqlite3")
    database.initialize()
    policy = OfficialNoticePolicy({
        name: adapter.config for name, adapter in load_source_registry(CONFIG).items()
    })
    return NoticeRepository(database, policy=policy)


def _item(**changes: object) -> dict[str, object]:
    item: dict[str, object] = {
        "id": "bksy-321714", "source_id": "undergraduate",
        "canonical_url": REAL_URL,
        "title": REAL_TITLE, "published_at": "2026-08-28",
        "deadline_at": "2026-09-06T17:30:00+08:00",
        "verified_at": "2026-09-25T16:55:00+08:00", "reviewer": "codex_reviewed",
        "summary": REAL_SUMMARY,
        "checks": CHECKS.copy(),
    }
    item.update(changes)
    return item


def _manifest(tmp_path: Path, entries: list[dict[str, object]]) -> Path:
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"schema_version": 1, "entries": entries},
                               ensure_ascii=False), encoding="utf-8")
    return path


def test_import_preserves_full_checked_summary_and_exact_deadline(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    item = _item()
    path = _manifest(tmp_path, [item])
    assert import_snapshot(path, repo).imported_count == 1
    assert import_snapshot(path, repo).unchanged_count == 1
    record = repo.list_verified()[0]
    assert record.summary == item["summary"]
    assert len(record.summary) > 160
    assert record.published_at == datetime(2026, 8, 27, 16, tzinfo=UTC)
    assert record.deadline_at == datetime(2026, 9, 6, 9, 30, tzinfo=UTC)
    assert record.deadline_confidence is DeadlineConfidence.HIGH
    assert record.proof_status is ProofStatus.VERIFIED_SNAPSHOT
    assert len(repo.detail_evidence(record.id)["versions"]) == 1


@pytest.mark.parametrize("change", [
    {"checks": {**CHECKS, "summary": False}},
    {"checks": {key: True for key in CHECKS if key != "sensitivity"}},
    {"summary": "联系 student@example.com 办理"},
    {"title": "[演示]课程安排通知"},
    {"canonical_url": "https://demo.invalid/info/1015/300001.htm"},
    {"source_id": "computer"},
    {"verified_at": "2999-01-01T00:00:00+08:00"},
    {"reviewer": "human"},
    {"summary": "  "},
    {"deadline_at": "2025-09-30T10:00:00"},
])
def test_invalid_manifest_rejects_before_any_write(tmp_path: Path, change: dict) -> None:
    repo = _repository(tmp_path)
    second = _item(id="second", canonical_url=OTHER_REAL_URL,
                   title=OTHER_REAL_TITLE,
                   summary="该通知说明本科生2026至2027学年秋季学期的选课安排；五月的两轮选课现已结束。",
                   published_at="2026-05-07")
    second.update(change)
    path = _manifest(tmp_path, [_item(), second])
    with pytest.raises(ValueError):
        import_snapshot(path, repo)
    assert repo.count() == 0


def test_duplicate_urls_reject_before_any_write(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    path = _manifest(tmp_path, [_item(), _item(id="second")])
    with pytest.raises(ValueError, match="duplicate"):
        import_snapshot(path, repo)
    assert repo.count() == 0


def test_import_date_only_deadline_keeps_low_confidence(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    item = _item(
        id="gs-173222", source_id="graduate",
        canonical_url="https://gs.shu.edu.cn/info/1029/173222.htm",
        title="上海大学2026级研究生新生学籍信息录入与培养计划登录通知",
        published_at="2026-09-11", deadline_at="2026-10-11",
        summary=("该通知面向2026级研究生新生，要求在2026年9月11日至10月11日期间登录研究生综合服务平台，"
                 "填写学籍信息并提交培养计划，打印入学登记表和培养计划交所在学院。"),
    )
    assert import_snapshot(_manifest(tmp_path, [item]), repo).imported_count == 1
    record = repo.find_by_url(str(item["canonical_url"]))
    assert record is not None
    assert record.deadline_at == datetime(2026, 10, 10, 16, tzinfo=UTC)
    assert record.deadline_confidence is DeadlineConfidence.LOW


def test_reviewed_computer_recommendation_notice_has_no_invented_deadline(
    tmp_path: Path,
) -> None:
    payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    entry = next(row for row in payload["entries"] if row["id"] == "cs-1081-52051")
    assert entry["source_id"] == "computer"
    assert entry["canonical_url"] == "https://cs.shu.edu.cn/info/1081/52051.htm"
    assert entry["title"] == "上海大学计算机工程与科学学院接收2027年推免研究生复试工作安排"
    assert entry["published_at"] == "2026-09-16"
    assert entry["deadline_at"] is None
    assert "分批" in entry["summary"]
    assert "邮件" in entry["summary"]
    assert "规定时间" not in entry["summary"]

    repo = _repository(tmp_path)
    assert import_snapshot(SNAPSHOT, repo).imported_count == len(payload["entries"])
    record = repo.find_by_url(entry["canonical_url"])
    assert record is not None
    assert record.deadline_at is None
    assert record.deadline_confidence is DeadlineConfidence.UNKNOWN
    assert record.proof_status is ProofStatus.VERIFIED_SNAPSHOT
    assert record.summary == entry["summary"]


def test_snapshot_does_not_downgrade_live_record(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    item = _item(summary="该通知面向2026级本科新生；两轮网上选课均已结束。")
    url = str(item["canonical_url"])
    live = Verification(ProofStatus.VERIFIED_LIVE,
                        datetime(2026, 9, 25, 8, 56, tzinfo=UTC), "live:checked-body")
    live_summary = "该通知规定新生选课学分上限为35学分，并要求正式上课前确认课表信息。"
    repo.upsert(NoticeDraft(source_id="undergraduate", canonical_url=url,
                            title=REAL_TITLE, normalized_text=live_summary),
                verification=live, summary_override=live_summary)
    path = _manifest(tmp_path, [item])
    assert import_snapshot(path, repo).unchanged_count == 1
    record = repo.find_by_url(url)
    assert record is not None
    assert (record.proof_status, record.verified_at, record.title, record.summary) == (
        ProofStatus.VERIFIED_LIVE, live.verified_at, REAL_TITLE, live_summary
    )
