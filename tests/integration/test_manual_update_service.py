from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import (
    ManualUpdateRunRepository,
    NoticeRepository,
    PendingCandidateRepository,
    SourceRepository,
)
from shu_insight.db.schema import Database
from shu_insight.ingest.http_client import PoliteHttpClient
from shu_insight.services.notices import NoticeService
from shu_insight.services.update import UpdateService
from shu_insight.sources.registry import load_source_registry

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 7, tzinfo=UTC)
CONTACT = "local-only-sensitive@example.invalid"


@pytest.fixture
def database(tmp_path):
    database = Database(tmp_path / "service.sqlite3")
    database.initialize()
    return database


def registry():
    return load_source_registry(ROOT / "config/sources.toml")


def service(database, factory, *, sources=None):
    return UpdateService(
        database=database, registry=registry() if sources is None else sources,
        client_factory=factory, today=lambda: NOW.date(), now=lambda: NOW,
    )


def local_factory(handler, clients):
    def create():
        client = PoliteHttpClient(contact=CONTACT, transport=httpx.MockTransport(handler),
                                  sleep=lambda _: None)
        clients.append(client)
        return client
    return create


def synthetic_listing(source_id: str) -> str:
    article = '<a href="/info/1/1.htm">[演示]虚构通知</a>'
    if source_id == "shu_main":
        article = '<a href="/info/1/1.htm"><p class="bt">[演示]虚构通知</p></a>'
        return ('<div class="ej_main"><div class="w1400"><div class="list"><ul><li>'
                f'{article}</li></ul></div></div></div>')
    if source_id == "graduate":
        return (f'<table width="755"><tr id="line_u17_0"><td>{article}</td>'
                '<td>2026/10/07</td></tr></table>')
    return ('<div class="tzgg"><ul><li><a href="/info/1/1.htm">'
            '<div class="tz-tx"><h3>[演示]虚构通知</h3></div>'
            '</a></li></ul></div>')


def test_full_registry_updates_only_two_sources_with_fresh_client_serial_order(database):
    sources = registry()
    calls = []
    clients = []
    hosts = {adapter.config.base_url.split("/")[2]: key for key, adapter in sources.items()}

    def handler(request):
        assert CONTACT in request.headers["User-Agent"]
        source = hosts[request.url.host]
        calls.append(source)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if source == "undergraduate":
            return httpx.Response(403)
        # Local fixture with only a synthetic title: proves successful parsing
        # without admitting or fetching a notice body.
        return httpx.Response(200, text=synthetic_listing(source),
                              headers={"Content-Type": "text/html"})

    sources["computer"] = type(sources["computer"])(replace(
        sources["computer"].config, reviewed_on=date(2026, 1, 1)))
    update = service(database, local_factory(handler, clients), sources=sources)
    events = []
    first = update.update_all(events.append)
    second = update.update_all()
    assert first.state == second.state == "partial"
    assert first.run_id and first.run_id != second.run_id
    assert [e.source_id for e in events if e.kind == "source_started"] == [
        "shu_main", "undergraduate"]
    assert first.by_source["shu_main"].state == "success"
    assert first.by_source["undergraduate"].state == "forbidden"
    assert set(first.by_source) == {"shu_main", "undergraduate"}
    assert calls == ["shu_main", "shu_main", "undergraduate", "undergraduate"] * 2
    assert len(clients) == 2 and clients[0] is not clients[1]
    assert all(c._client.is_closed for c in clients)
    with database.connect() as connection:
        rows = connection.execute("SELECT run_id, state FROM source_status ORDER BY id").fetchall()
        assert [r[0] for r in rows] == [first.run_id] * 2 + [second.run_id] * 2
        runs = connection.execute("SELECT * FROM update_runs ORDER BY rowid").fetchall()
        assert all(r["state"] == "partial" and r["finished_at"] == NOW.isoformat() for r in runs)
    assert ManualUpdateRunRepository(database).latest()["id"] == second.run_id


@pytest.mark.parametrize("failure", ["factory", "observer", "close"])
def test_exceptions_are_safe_terminal_and_release_lock(database, failure, caplog):
    class BadCloseClient(PoliteHttpClient):
        def close(self):
            super().close()
            if failure == "close":
                raise RuntimeError(CONTACT)

    def factory():
        if failure == "factory":
            raise RuntimeError(CONTACT)
        return BadCloseClient(contact=CONTACT, transport=httpx.MockTransport(
            lambda _: httpx.Response(403)), sleep=lambda _: None)

    def observer(event):
        if failure == "observer":
            raise RuntimeError(CONTACT)

    update = service(database, factory)
    report = update.update_all(observer)
    assert report.state == "failed"
    assert report.message == "update_failed"
    assert ManualUpdateRunRepository(database).latest()["state"] == "failed"
    assert CONTACT not in json.dumps(asdict(report))
    assert CONTACT not in caplog.text
    with database.connect() as connection:
        dump = "\n".join(connection.iterdump())
    assert CONTACT not in dump
    assert CONTACT.encode() not in database.path.read_bytes()
    retry = service(database, local_factory(lambda _: httpx.Response(403), [])).update_all()
    assert retry.run_id != report.run_id
    assert retry.state == "failed"
    assert SourceRepository(database).latest_statuses()


def test_all_success_aggregates_success(database):
    sources = {"shu_main": registry()["shu_main"]}
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, text=synthetic_listing("shu_main"),
                              headers={"Content-Type": "text/html"})
    report = service(database, local_factory(handler, []), sources=sources).update_all()
    assert report.state == "success"
    assert ManualUpdateRunRepository(database).latest()["state"] == "success"


def test_manual_run_keeps_rejected_and_url_only_candidates_out_of_official_feed(database):
    sources = {"shu_main": registry()["shu_main"]}
    base = "https://www.shu.edu.cn"
    attachment = f"{base}/info/1029/101.htm"
    redirect = f"{base}/info/1029/102.htm"
    bare = f"{base}/info/1029/999.htm"
    rejected_attachment = f"{base}/info/1029/103.pdf"
    rejected_host = "https://other.invalid/info/1029/104.htm"
    queue = PendingCandidateRepository(
        database, {key: item.config for key, item in sources.items()},
    )
    assert queue.enqueue("shu_main", bare, now=NOW)
    requests = []

    def handler(request):
        requests.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if str(request.url) == sources["shu_main"].config.list_url:
            links = (
                (attachment, "课程安排附件通知"),
                (redirect, "研究生办事流程通知"),
                (rejected_attachment, "附件下载通知"),
                (rejected_host, "站外通知"),
            )
            rows = "".join(
                f'<li><a href="{url}"><p class="bt">{title}</p>'
                '<p class="sj">2026-10-07</p></a></li>'
                for url, title in links
            )
            return httpx.Response(200, text=(
                '<div class="ej_main"><div class="w1400"><div class="list"><ul>'
                f'{rows}</ul></div></div></div>'),
                                  headers={"Content-Type": "text/html"})
        if str(request.url) == attachment:
            return httpx.Response(200, content=b"%PDF-1.7 fixture",
                                  headers={"Content-Type": "application/pdf"})
        if str(request.url) == redirect:
            return httpx.Response(302, headers={"Location": "https://other.invalid/notice"})
        pytest.fail(f"unexpected fixture URL: {request.url}")

    report = service(database, local_factory(handler, []), sources=sources).update_all()
    policy = OfficialNoticePolicy({key: item.config for key, item in sources.items()})
    notices = NoticeRepository(database, policy=policy)

    assert report.by_source["shu_main"].state == "redirect_not_allowed"
    assert report.by_source["shu_main"].queued_count == 1
    assert requests == [f"{base}/robots.txt", sources["shu_main"].config.list_url,
                        attachment, redirect]
    assert notices.list_browseable() == []
    assert NoticeService(notices, authority_by_source_id={}).all() == []
    assert queue.urls("shu_main") == {bare}


def start_process(bounded_child, database):
    code = """
import sys
from pathlib import Path
import httpx
import shu_insight.services.update as module
from shu_insight.db.schema import Database
from shu_insight.ingest.http_client import PoliteHttpClient
from shu_insight.sources.registry import load_source_registry
assert Path(module.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve() / 'src')
def factory():
    print('entered', flush=True)
    sys.stdin.readline()
    return PoliteHttpClient(contact='offline@example.invalid',
        transport=httpx.MockTransport(lambda _: httpx.Response(403)), sleep=lambda _: None)
service = module.UpdateService(database=Database(Path(sys.argv[1])),
    registry=load_source_registry(Path(sys.argv[2]) / 'config/sources.toml'),
    client_factory=factory)
report = service.update_all()
print(report.state, flush=True)
"""
    return bounded_child(
        [sys.executable, "-u", "-c", code, str(database.path), str(ROOT)],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


@pytest.mark.parametrize("terminate", [False, True])
def test_process_conflict_has_no_client_or_status_and_next_run_recovers(
    database, terminate, bounded_child,
):
    # Missing factory-only API fails in the parent before starting a child.
    update = service(database, lambda: pytest.fail("conflict must not construct a client"))
    with start_process(bounded_child, database) as first:
        first.wait_for_line("entered")
        with database.connect() as connection:
            abandoned_id = connection.execute("SELECT id FROM update_runs").fetchone()[0]
            assert connection.execute("SELECT COUNT(*) FROM source_status").fetchone()[0] == 0
        report = update.update_all(lambda _: pytest.fail("conflict must not emit source checks"))
        assert report.state == "concurrent_skipped"
        assert report.run_id is None and report.by_source == {} and report.events == ()
        with database.connect() as connection:
            assert connection.execute("SELECT COUNT(*) FROM update_runs").fetchone()[0] == 1
            assert connection.execute("SELECT COUNT(*) FROM source_status").fetchone()[0] == 0
        if terminate:
            first.process.kill()
        else:
            first.process.stdin.write("release\n")
            first.process.stdin.flush()
        _, errors = first.communicate()
        if not terminate:
            assert first.process.returncode == 0, errors
        if terminate:
            assert ManualUpdateRunRepository(database).latest()["state"] == "interrupted"
            with database.connect() as connection:
                row = connection.execute("SELECT finished_at FROM update_runs").fetchone()
                assert row[0] is None
        retry = service(database, local_factory(lambda _: httpx.Response(403), [])).update_all()
        assert retry.run_id != abandoned_id
        with database.connect() as connection:
            old = connection.execute("SELECT state, finished_at FROM update_runs WHERE id = ?",
                                     (abandoned_id,)).fetchone()
            assert old[0] == ("interrupted" if terminate else "failed")
            assert old[1] is not None


def test_noneligible_registry_does_not_construct_client_or_write_run(database):
    calls = []
    report = service(database, lambda: calls.append(True),
                     sources={"graduate": registry()["graduate"]}).update_all()
    assert report.state == "failed" and report.message == "no_eligible_sources"
    assert report.by_source == {} and calls == []
    assert ManualUpdateRunRepository(database).latest() is None
