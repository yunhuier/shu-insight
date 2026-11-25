"""Offline preload admission and preservation regressions."""

import importlib
import json
import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository
from shu_insight.db.schema import Database
from shu_insight.domain.models import (
    DiscoveredNotice,
    NoticeDraft,
    Persona,
    ProofStatus,
    Verification,
)
from shu_insight.services.notices import NoticeQuery, NoticeService
from shu_insight.sources.registry import load_source_registry

ROOT = Path(__file__).resolve().parents[2]
TIME = datetime(2026, 10, 12, 12, tzinfo=UTC)
URL = "https://bksy.shu.edu.cn/info/1015/900001.htm"
ATTACHMENT_URL = "https://bksy.shu.edu.cn/info/1015/322264.htm"


def attachment_exclusion():
    return {
        'source_id': 'undergraduate', 'canonical_url': ATTACHMENT_URL,
        'sha256': '1566304fe3b57761c02fed7d574212cf9ae73f3cd5deccfeedfcb5d082e0e79b',
        'reviewed_at': TIME.isoformat(), 'reviewer': 'codex_reviewed',
        'checks': {'url': True, 'attachment_only': True, 'sensitivity': True},
    }


@pytest.fixture
def registry():
    return load_source_registry(ROOT / "config/sources.toml")


@pytest.fixture
def repository(tmp_path, registry):
    database = Database(tmp_path / "source.sqlite3")
    database.initialize()
    return NoticeRepository(database, OfficialNoticePolicy({
        key: adapter.config for key, adapter in registry.items()
    }))


def api():
    assert importlib.util.find_spec("shu_insight.ingest.preload") is not None, (
        "offline preload implementation is missing"
    )
    return importlib.import_module("shu_insight.ingest.preload")


def seed(repository, *, listing=False, summary="请同学们通过学校平台提交申请材料。"):
    if listing:
        repository.upsert_listing(
            DiscoveredNotice("undergraduate", URL, "公开申报通知", "2026-10-11"),
            list_url="https://bksy.shu.edu.cn/index/tzgg.htm", checked_at=TIME,
        )
    else:
        repository.upsert(NoticeDraft("undergraduate", URL, "公开申报通知", summary),
                          verification=Verification(ProofStatus.VERIFIED_LIVE, TIME,
                                                    "live:" + "a" * 64))


def export(repository, tmp_path, registry):
    target = tmp_path / "preload.json"
    api().export_preload(repository.database.path, target, registry)
    return target


def test_export_is_readonly_and_import_is_idempotent(repository, tmp_path, registry):
    seed(repository)
    before = repository.database.path.read_bytes()
    manifest = export(repository, tmp_path, registry)
    assert repository.database.path.read_bytes() == before
    destination = Database(tmp_path / "destination.sqlite3")
    destination.initialize()
    target = NoticeRepository(destination, repository.policy)
    first = api().import_preload(manifest, target)
    second = api().import_preload(manifest, target)
    assert (first.imported_count, first.rejected_count, second.unchanged_count) == (1, 0, 1)
    original = repository.find_by_url(URL)
    restored = target.find_by_url(URL)
    assert (restored.id, restored.fetched_at, restored.verified_at, restored.verification_ref) == (
        original.id, original.fetched_at, TIME, "live:" + "a" * 64,
    )
    with destination.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM notice_versions").fetchone()[0] == 1


def test_listing_never_exports_legacy_body_or_deadlines(repository, tmp_path, registry):
    seed(repository, listing=True)
    with repository.database.transaction() as connection:
        connection.execute("UPDATE notices SET summary='legacy private body', "
                           "deadline_at='2026-11-01T00:00:00+00:00', deadline_confidence='high'")
    payload = json.loads(export(repository, tmp_path, registry).read_text(encoding="utf-8"))
    entry = payload["entries"][0]
    assert (entry["summary"], entry["deadline_at"], entry["deadline_confidence"]) == (
        "", None, "unknown",
    )


@pytest.mark.parametrize("field,value", [
    ("canonical_url", "https://example.org/info/1015/900001.htm"),
    ("canonical_url", "https://bksy.shu.edu.cn/info/1015/900001.pdf"),
    ("title", "[演示]公开申报通知"),
    ("proof_status", "unverified_legacy"),
    ("verification_ref", "live:untrusted"),
    ("summary", "研究生张三同学获得国家奖学金。"),
    ("source_id", "shu_main"),
])
def test_import_rejects_untrusted_entries(repository, tmp_path, registry, field, value):
    seed(repository)
    manifest = export(repository, tmp_path, registry)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["entries"][0][field] = value
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    report = api().import_preload(manifest, repository)
    assert report.rejected_count == 1
    assert repository.find_by_url(URL).summary == "请同学们通过学校平台提交申请材料。"


def test_export_reapplies_privacy_filter(repository, tmp_path, registry):
    seed(repository, summary="研究生张三同学获得国家奖学金。请同学们通过学校平台提交申请材料。")
    content = export(repository, tmp_path, registry).read_text(encoding="utf-8")
    assert "张三" not in content
    assert "请同学们通过学校平台提交申请材料" in content


@pytest.mark.parametrize("protected", ["newer", "parsed", "attachment"])
def test_import_preserves_higher_quality_newer_and_excluded(repository, tmp_path, registry,
                                                         protected):
    seed(repository, listing=protected == "parsed")
    manifest = export(repository, tmp_path, registry)
    if protected == "attachment":
        with repository.database.transaction() as connection:
            connection.execute("UPDATE notices SET proof_status='attachment_excluded'")
    else:
        repository.upsert(NoticeDraft("undergraduate", URL, "较新的公开通知",
                                     "请按新流程提交材料。"),
                          verification=Verification(ProofStatus.VERIFIED_LIVE,
                                                    datetime(2026, 10, 20, tzinfo=UTC),
                                                    "live:" + "b" * 64))
    before = repository.find_by_url(URL)
    report = api().import_preload(manifest, repository)
    assert report.unchanged_count == 1
    assert repository.find_by_url(URL) == before


def test_export_omits_demo_and_attachment_tombstones(repository, tmp_path, registry):
    seed(repository, listing=True)
    repository.withdraw_listing(URL)
    payload = json.loads(export(repository, tmp_path, registry).read_text(encoding="utf-8"))
    assert payload["entries"] == []


def test_import_does_not_open_http(repository, tmp_path, registry, monkeypatch):
    import socket

    seed(repository)
    manifest = export(repository, tmp_path, registry)

    def forbidden(*args, **kwargs):
        raise AssertionError("offline preload attempted network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    assert api().import_preload(manifest, repository).unchanged_count == 1


@pytest.mark.parametrize("script", ["export_official_preload.py", "import_official_preload.py"])
def test_preload_cli_requires_explicit_manifest_and_database(script):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script)],
                            capture_output=True, text=True, timeout=20,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert result.returncode == 2
    assert "--manifest" in result.stderr and "--database" in result.stderr


def test_reviewed_computer_excerpt_excludes_working_group_names():
    from bs4 import BeautifulSoup

    soup = BeautifulSoup('<div class="v_news_content"><p class="vsbcontent_start">'
                         '根据学校办法，特制订本实施办法。</p><table><tr><td>9月4-6日</td>'
                         '<td>学生报名；学院接收学生的报名材料（交到年级辅导员处）</td></tr>'
                         '</table><p>组长：张三、李四</p></div>', 'lxml')
    assert api()._reviewed_computer_preview(soup) == (
        '根据学校办法，特制订本实施办法。9月4-6日 学生报名；'
        '学院接收学生的报名材料（交到年级辅导员处）'
    )


def test_pagination_listing_proof_survives_without_inventing_its_page_url(repository, tmp_path,
                                                                        registry):
    repository.upsert_listing(
        DiscoveredNotice("undergraduate", URL, "关于整理推荐名单的通知", "2026-10-11"),
        list_url="https://bksy.shu.edu.cn/index/tzgg/101.htm", checked_at=TIME,
    )
    original = repository.find_by_url(URL)
    manifest = export(repository, tmp_path, registry)
    target_db = Database(tmp_path / "pagination.sqlite3")
    target_db.initialize()
    target = NoticeRepository(target_db, repository.policy)
    assert api().import_preload(manifest, target).imported_count == 1
    record = target.find_by_url(URL)
    assert (record.title, record.verification_ref, record.summary) == (
        "关于整理推荐名单的通知", original.verification_ref, "",
    )


def test_import_cli_refuses_schema_migration(tmp_path, repository, registry):
    seed(repository)
    manifest = export(repository, tmp_path, registry)
    legacy = tmp_path / 'legacy.sqlite3'
    with sqlite3.connect(legacy) as connection:
        connection.execute('CREATE TABLE schema_meta(version INTEGER)')
        connection.execute('INSERT INTO schema_meta VALUES (1)')
    before = legacy.read_bytes()
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/import_official_preload.py'),
         '--database', str(legacy), '--manifest', str(manifest)],
        capture_output=True, text=True, timeout=20,
        env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')},
    )
    assert result.returncode == 2
    assert legacy.read_bytes() == before
    assert not (tmp_path / 'backups').exists()


def test_parsed_metadata_roundtrip_preserves_persona_only_recommendations(repository, tmp_path,
                                                                       registry):
    repository.upsert(NoticeDraft(
        'undergraduate', URL, '公开申报通知', '请按流程提交材料。',
        education_levels=('本科生',), grades=('大三',), colleges=('计算机工程与科学学院',),
        majors=('计算机科学与技术',), categories=('科研学术',), topic_tags=('科研',),
    ), verification=Verification(ProofStatus.VERIFIED_LIVE, TIME, 'live:' + 'a' * 64))
    manifest = export(repository, tmp_path, registry)
    destination = Database(tmp_path / 'persona.sqlite3')
    destination.initialize()
    target = NoticeRepository(destination, repository.policy)
    assert api().import_preload(manifest, target).imported_count == 1
    record = target.find_by_url(URL)
    assert (record.education_levels, record.grades, record.colleges, record.majors,
            record.categories, record.topic_tags) == (
        ('本科生',), ('大三',), ('计算机工程与科学学院',), ('计算机科学与技术',),
        ('科研学术',), ('科研',),
    )
    service = NoticeService(target, authority_by_source_id={'undergraduate': .9}, now=lambda: TIME)
    matched = service.recommend(NoticeQuery(persona=Persona(
        'profile', '我的画像', education_level='本科生', grade='大三',
        college='计算机工程与科学学院', major='计算机科学与技术',
    )))
    assert len(matched) == 1
    assert matched[0].match_tier == 'possible'
    assert '适用于计算机科学与技术' in matched[0].reasons


def test_listing_discards_legacy_audience_metadata(repository, tmp_path, registry):
    seed(repository, listing=True)
    with repository.database.transaction() as connection:
        connection.execute('UPDATE notices SET education_levels_json=?, grades_json=?, '
                           'colleges_json=?, majors_json=?, categories_json=?, topic_tags_json=?',
                           tuple(json.dumps(value) for value in (
                               ['本科生'], ['大三'], ['计算机工程与科学学院'], ['计算机科学与技术'],
                               ['科研学术'], ['科研'],
                           )))
    manifest = export(repository, tmp_path, registry)
    payload = json.loads(manifest.read_text(encoding='utf-8'))
    entry = payload['entries'][0]
    for field in ('education_levels', 'grades', 'colleges', 'majors', 'categories', 'topic_tags'):
        assert entry[field] == []
    destination = Database(tmp_path / 'listing-persona.sqlite3')
    destination.initialize()
    target = NoticeRepository(destination, repository.policy)
    assert api().import_preload(manifest, target).imported_count == 1
    service = NoticeService(target, authority_by_source_id={'undergraduate': .9}, now=lambda: TIME)
    matched = service.recommend(NoticeQuery(persona=Persona(
        'profile', '我的画像', education_level='本科生', grade='大三',
        college='计算机工程与科学学院', major='计算机科学与技术',
    )))
    assert len(matched) == 1
    assert matched[0].match_tier == 'other'
    assert matched[0].reasons == ()


def test_import_rejects_untrusted_personal_metadata(repository, tmp_path, registry):
    seed(repository)
    manifest = export(repository, tmp_path, registry)
    payload = json.loads(manifest.read_text(encoding='utf-8'))
    payload['entries'][0]['colleges'] = ['张三老师办公室']
    manifest.write_text(json.dumps(payload), encoding='utf-8')
    assert api().import_preload(manifest, repository).rejected_count == 1


@pytest.mark.parametrize('tail', [
    '联系人：张三，电话：66133507-803', '联系方式 李老师 66130833*8001',
    '组长：张三、李四', '办公电话：66131606*801',
])
def test_export_removes_flattened_contact_and_roster_tails(repository, tmp_path, registry, tail):
    seed(repository, summary='请按流程提交申请材料。' + tail)
    content = export(repository, tmp_path, registry).read_text(encoding='utf-8')
    assert '张三' not in content and '李老师' not in content
    assert not any(number in content for number in ('66133507', '66130833', '66131606'))
    assert '请按流程提交申请材料' in content


def test_export_masks_unlabelled_landline_in_public_procedure(repository, tmp_path, registry):
    seed(repository, summary='材料流程咨询请拨打66133507-803。')
    content = export(repository, tmp_path, registry).read_text(encoding='utf-8')
    assert '66133507' not in content
    assert '材料流程咨询' in content


def test_import_requires_literal_true_offline_review_checks(repository, tmp_path, registry):
    seed(repository)
    manifest = export(repository, tmp_path, registry)
    payload = json.loads(manifest.read_text(encoding='utf-8'))
    entry = payload['entries'][0]
    entry.update(proof_status='verified_snapshot', fetched_at=None,
                 verification_ref='snapshot:' + 'a' * 64)
    entry['provenance'] = {
        'kind': 'offline_article_review', 'sha256': 'a' * 64,
        'capture_time_basis': 'unknown_saved_file_capture_time', 'reviewer': 'codex_reviewed',
        'checks': dict.fromkeys(('url', 'title', 'date', 'audience', 'action', 'summary',
                                'sensitivity'), 1),
    }
    manifest.write_text(json.dumps(payload), encoding='utf-8')
    assert api().import_preload(manifest, repository).rejected_count == 1


def test_export_omits_reviewed_attachment_listing_and_carries_provenance(repository, tmp_path,
                                                                         registry):
    repository.upsert_listing(
        DiscoveredNotice('undergraduate', ATTACHMENT_URL, '项目申报通知'),
        list_url='https://bksy.shu.edu.cn/index/tzgg.htm', checked_at=TIME,
    )
    manifest = tmp_path / 'reviewed-exclusion.json'
    api().export_preload(repository.database.path, manifest, registry,
                         reviewed_exclusions=[attachment_exclusion()])
    payload = json.loads(manifest.read_text(encoding='utf-8'))
    assert payload['entries'] == []
    assert payload['attachment_exclusions'] == [attachment_exclusion()]


@pytest.mark.parametrize('parsed', [False, True])
def test_reviewed_attachment_withdraws_existing_listing_but_preserves_parsed(
    repository, tmp_path, registry, parsed,
):
    if parsed:
        repository.upsert(
            NoticeDraft('undergraduate', ATTACHMENT_URL, '项目申报通知', '已读取的新正文。'),
            verification=Verification(ProofStatus.VERIFIED_LIVE,
                                      datetime(2026, 10, 20, tzinfo=UTC), 'live:' + 'a' * 64),
        )
    else:
        repository.upsert_listing(
            DiscoveredNotice('undergraduate', ATTACHMENT_URL, '项目申报通知'),
            list_url='https://bksy.shu.edu.cn/index/tzgg.htm', checked_at=TIME,
        )
    before = repository.find_by_url(ATTACHMENT_URL)
    with repository.database.connect() as connection:
        versions = list(connection.execute('SELECT * FROM notice_versions'))
    manifest = tmp_path / 'withdraw.json'
    manifest.write_text(json.dumps({
        'schema_version': 1, 'kind': 'official_offline_preload', 'entries': [],
        'attachment_exclusions': [attachment_exclusion()],
    }), encoding='utf-8')
    report = api().import_preload(manifest, repository)
    after = repository.find_by_url(ATTACHMENT_URL)
    if parsed:
        assert after == before
        assert report.withdrawn_count == 0
    else:
        assert after.id == before.id
        assert after.proof_status == 'attachment_excluded'
        assert after.verified_at is None and not after.verification_ref
        assert repository.list_browseable() == []
        assert report.withdrawn_count == 1
        assert api().import_preload(manifest, repository).withdrawn_count == 0
    with repository.database.connect() as connection:
        assert [tuple(row) for row in connection.execute('SELECT * FROM notice_versions')] == [
            tuple(row) for row in versions
        ]


@pytest.mark.parametrize('field,value', [
    ('canonical_url', 'https://bksy.shu.edu.cn/info/1015/900001.htm'),
    ('source_id', 'shu_main'), ('sha256', '0' * 64),
    ('checks', {'url': 1, 'attachment_only': True, 'sensitivity': True}),
])
def test_unproven_attachment_withdrawal_is_rejected(repository, tmp_path, field, value):
    repository.upsert_listing(
        DiscoveredNotice('undergraduate', ATTACHMENT_URL, '项目申报通知'),
        list_url='https://bksy.shu.edu.cn/index/tzgg.htm', checked_at=TIME,
    )
    exclusion = attachment_exclusion()
    exclusion[field] = value
    manifest = tmp_path / 'unproven.json'
    manifest.write_text(json.dumps({
        'schema_version': 1, 'kind': 'official_offline_preload', 'entries': [],
        'attachment_exclusions': [exclusion],
    }), encoding='utf-8')
    before = repository.find_by_url(ATTACHMENT_URL)
    assert api().import_preload(manifest, repository).rejected_count == 1
    assert repository.find_by_url(ATTACHMENT_URL) == before
