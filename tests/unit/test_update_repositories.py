from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.compliance.policy import evaluate_response
from shu_insight.db import repositories
from shu_insight.db.schema import Database
from shu_insight.domain.models import NoticeDraft, ProofStatus, SourceConfig, Verification
from shu_insight.sources.registry import load_source_registry

NOW = datetime(2026, 10, 6, tzinfo=UTC)
URL = "https://cs.shu.edu.cn/info/1/1.htm"


@pytest.fixture
def database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "update.sqlite3")
    database.initialize()
    return database


@pytest.fixture
def sources() -> dict[str, SourceConfig]:
    return {key: adapter.config for key, adapter in
            load_source_registry(Path("config/sources.toml")).items()}


def queue(database: Database, sources: dict[str, SourceConfig]):
    return repositories.PendingCandidateRepository(database, sources)


def test_v5_status_counters_roundtrip_and_reject_non_numeric_private_payloads(database):
    repo = repositories.SourceRepository(database)
    repo.record_status({
        "source_id": "graduate", "state": "partial", "discovered_count": 45,
        "parsed_count": 31, "listing_count": 14, "pending_count": 17, "queued_count": 3,
        "request_breakdown": {"robots": 1, "list": 5, "detail": 31, "retry": 3,
                              "private@example.invalid": "<html>"},
    })
    result, = repo.latest_statuses()
    assert {key: result[key] for key in (
        "discovered_count", "parsed_count", "listing_count", "pending_count", "queued_count",
    )} == {"discovered_count": 45, "parsed_count": 31, "listing_count": 14,
          "pending_count": 17, "queued_count": 3}
    assert result["request_breakdown"] == {"robots": 1, "list": 5, "detail": 31, "retry": 3}
    repo.record_status({
        "source_id": "graduate", "state": "partial", "discovered_count": "private text",
        "parsed_count": -9, "listing_count": True, "pending_count": 10**90,
        "request_breakdown": {"robots": "private", "list": -1, "detail": 10**90,
                              "retry": True},
    })
    result, = repo.latest_statuses()
    assert result["discovered_count"] == result["parsed_count"] == result["listing_count"] == 0
    assert result["pending_count"] == 10000
    assert result["request_breakdown"] == {"robots": 0, "list": 0, "detail": 40, "retry": 0}
    with database.transaction() as connection:
        row = connection.execute(
            "SELECT request_breakdown_json FROM source_status LIMIT 1",
        ).fetchone()
        assert "private" not in row[0] and "html" not in row[0]
        connection.execute("UPDATE source_status SET request_breakdown_json = ?",
                           ('{"robots":true,"list":-1,"detail":999999,"raw":"secret"}',))
    assert repo.latest_statuses()[0]["request_breakdown"] == {
        "robots": 0, "list": 0, "detail": 40, "retry": 0,
    }


def test_run_latest_is_read_only_and_unfinished_is_interrupted(database: Database) -> None:
    repo = repositories.ManualUpdateRunRepository(database)
    assert repo.latest() is None
    run_id = repo.start(started_at=NOW)
    assert repo.latest() == {
        "id": run_id, "started_at": NOW.isoformat(), "finished_at": None,
        "state": "interrupted",
    }
    with database.connect() as connection:
        row = connection.execute("SELECT state, finished_at FROM update_runs").fetchone()
        assert tuple(row) == ("running", None)
    repo.finish(run_id, "partial", finished_at=NOW + timedelta(minutes=1))
    assert repo.latest()["state"] == "partial"
    assert repo.latest()["finished_at"] == "2026-10-06T00:01:00+00:00"


def test_start_after_lock_marks_old_run_interrupted(database: Database) -> None:
    repo = repositories.ManualUpdateRunRepository(database)
    old_id = repo.start(started_at=NOW)
    new_id = repo.start(started_at=NOW + timedelta(minutes=1))
    assert new_id != old_id
    with database.connect() as connection:
        row = connection.execute("SELECT state, finished_at FROM update_runs WHERE id = ?",
                                 (old_id,)).fetchone()
        assert tuple(row) == ("interrupted", "2026-10-06T00:01:00+00:00")
    assert repo.latest()["id"] == new_id


def test_run_finish_rejects_raw_text_and_is_terminal(database: Database) -> None:
    repo = repositories.ManualUpdateRunRepository(database)
    run_id = repo.start(started_at=NOW)
    repo.finish(run_id, "<html>private@example.invalid</html>", finished_at=NOW)
    repo.finish(run_id, "success", finished_at=NOW + timedelta(minutes=1))
    assert repo.latest()["state"] == "failed"
    assert repo.latest()["finished_at"] == NOW.isoformat()


def test_source_history_separates_list_success_from_completed_check(database: Database) -> None:
    repo = repositories.SourceRepository(database)
    run_repo = repositories.ManualUpdateRunRepository(database)
    complete_id = run_repo.start(started_at=NOW)
    repo.record_status({
        "source_id": "computer", "state": "success", "checked_at": NOW,
        "run_id": complete_id, "list_pages_read": 1, "request_count": 4,
        "queued_count": 0, "coverage_code": "pagination_unverified",
    })
    run_repo.finish(complete_id, "success", finished_at=NOW)
    partial_time = NOW + timedelta(days=1)
    partial_id = run_repo.start(started_at=partial_time)
    repo.record_status({
        "source_id": "computer", "state": "partial", "checked_at": partial_time,
        "run_id": partial_id, "list_pages_read": 2, "request_count": 20,
        "queued_count": 3, "coverage_code": "request_budget_exhausted",
    })
    run_repo.finish(partial_id, "partial", finished_at=partial_time)
    repo.record_status({"source_id": "computer", "state": "error",
                        "checked_at": NOW + timedelta(days=2)})
    assert repo.latest_statuses()[0]["state"] == "error"
    successful_list = repo.latest_successful_list_checks()[0]
    assert successful_list["checked_at"] == partial_time.isoformat()
    assert successful_list["run_id"] == partial_id
    assert successful_list["queued_count"] == 3
    completed = repo.latest_completed_checks()[0]
    assert completed["checked_at"] == NOW.isoformat()
    assert completed["run_id"] == complete_id
    assert completed["coverage_code"] == "pagination_unverified"


@pytest.mark.parametrize("message,want", [
    ("forbidden", "forbidden"), ("robots_disallowed", "robots_disallowed"),
    ("robots_unavailable", "robots_unavailable"),
    ("contact=private@example.invalid", "partial"),
])
def test_partial_source_reason_roundtrip_preserves_only_safe_codes(database, message, want):
    repo = repositories.SourceRepository(database)
    repo.record_status({"source_id": "graduate", "state": "partial", "message": message})
    assert repo.latest_statuses()[0]["message"] == want
    with database.connect() as connection:
        assert connection.execute("SELECT message FROM source_status").fetchone()[0] == want


def test_partial_legacy_reason_read_rejects_raw_contact(database):
    repo = repositories.SourceRepository(database)
    repo.record_status({"source_id": "graduate", "state": "partial"})
    with database.transaction() as connection:
        connection.execute("UPDATE source_status SET message = 'robots_disallowed'")
    assert repo.latest_statuses()[0]["message"] == "robots_disallowed"
    with database.transaction() as connection:
        connection.execute("UPDATE source_status SET message = 'private@example.invalid'")
    assert repo.latest_statuses()[0]["message"] == "partial"


@pytest.mark.parametrize("stage", ["robots", "list", "detail"])
@pytest.mark.parametrize("kind", [
    "proxy", "connect", "connect_timeout", "read", "read_timeout", "other",
    "remote_protocol", "local_protocol", "write", "write_timeout", "pool_timeout",
    "close", "unsupported_protocol",
])
def test_source_status_roundtrips_only_enumerated_transport_code(database, stage, kind):
    code = f"transport_{stage}_{kind}"
    repo = repositories.SourceRepository(database)
    repo.record_status({"source_id": "graduate", "state": "partial", "message": code})
    with database.connect() as connection:
        assert connection.execute("SELECT message FROM source_status").fetchone()[0] == code
    assert repo.latest_statuses()[0]["message"] == code


@pytest.mark.parametrize("unsafe", [
    "transport_detail_proxy_http://proxy.invalid:8080",
    "transport_list_read_timeout_private@example.invalid",
    "transport_robots_dns", "transport_login_proxy",
    "transport_detail_remote_protocol_private@example.invalid",
    "transport_list_write_timeout_http://user:password@proxy.invalid",
    "transport_robots_pool_timeout_synthetic-transport-secret",
])
def test_source_status_rejects_unenumerated_transport_details_on_write_and_read(
    database, unsafe,
):
    repo = repositories.SourceRepository(database)
    repo.record_status({"source_id": "computer", "state": "connection_failed",
                        "message": unsafe})
    with database.connect() as connection:
        assert connection.execute("SELECT message FROM source_status").fetchone()[0] == (
            "connection_failed"
        )
    with database.transaction() as connection:
        connection.execute("UPDATE source_status SET message = ?", (unsafe,))
    assert repo.latest_statuses()[0]["message"] == "connection_failed"


@pytest.mark.parametrize(("status", "final_url", "expected"), [
    (300, URL, "unresolved_redirect"),
    (200, "https://cs.shu.edu.cn:bad/info/1/1.htm", "invalid_url"),
    (200, "http://cs.shu.edu.cn/info/1/1.htm", "https_required"),
    (200, "https://other.invalid/info/1/1.htm", "host_not_allowed"),
])
@pytest.mark.parametrize("partial", [False, True])
def test_fixed_policy_failure_survives_source_status_write_and_read(
    database, sources, status, final_url, expected, partial,
):
    decision = evaluate_response(sources["computer"], status, "text/html", final_url,
                                 "<h1>公开课程通知</h1>")
    assert not decision.allowed and decision.reason == expected
    state = "partial" if partial else decision.reason
    repositories.SourceRepository(database).record_status({
        "source_id": "computer", "state": state, "message": decision.reason,
    })
    with database.connect() as connection:
        row = connection.execute("SELECT state, message FROM source_status").fetchone()
        assert tuple(row) == (state, expected)
    result = repositories.SourceRepository(database).latest_statuses()[0]
    assert (result["state"], result["message"]) == (state, expected)


@pytest.mark.parametrize("reason", [
    "unresolved_redirect", "invalid_url", "https_required", "host_not_allowed",
])
def test_legacy_fixed_policy_failure_remains_precise_when_read(database, reason):
    repo = repositories.SourceRepository(database)
    repo.record_status({"source_id": "computer", "state": "error"})
    with database.transaction() as connection:
        connection.execute("UPDATE source_status SET state = ?, message = ?", (reason, reason))
    result = repo.latest_statuses()[0]
    assert (result["state"], result["message"]) == (reason, reason)


@pytest.mark.parametrize("raw_reason", [
    "unresolved_redirect contact=private@example.invalid",
    "https_required:<html>body</html>", "UNRESOLVED_REDIRECT",
])
def test_policy_reason_admission_does_not_accept_raw_or_unknown_text(database, raw_reason):
    repo = repositories.SourceRepository(database)
    repo.record_status({"source_id": "computer", "state": raw_reason, "message": raw_reason})
    with database.connect() as connection:
        row = connection.execute("SELECT state, message FROM source_status").fetchone()
        assert tuple(row) == ("error", "error")
    with database.transaction() as connection:
        connection.execute("UPDATE source_status SET state = ?, message = ?",
                           (raw_reason, raw_reason))
    result = repo.latest_statuses()[0]
    assert (result["state"], result["message"]) == ("error", "error")


def test_legacy_status_readable_but_does_not_claim_list_parse(database: Database) -> None:
    repo = repositories.SourceRepository(database)
    repo.record_status({"source_id": "computer", "state": "success", "checked_at": NOW,
                        "list_pages_read": 99})
    status = repo.latest_statuses()[0]
    assert status["run_id"] is None
    assert status["list_pages_read"] == 0
    assert repo.latest_successful_list_checks() == []
    assert repo.latest_completed_checks()[0]["checked_at"] == NOW.isoformat()


def test_successful_list_history_excludes_offline_seed(database: Database) -> None:
    repo = repositories.SourceRepository(database)
    run_id = repositories.ManualUpdateRunRepository(database).start(started_at=NOW)
    repo.record_status({"source_id": "computer", "state": "success", "run_id": run_id,
                        "checked_at": NOW, "list_pages_read": 1})
    with database.transaction() as connection:
        connection.execute("UPDATE source_status SET message = 'synthetic_demo: offline'")
    assert repo.latest_successful_list_checks() == []
    assert repo.latest_completed_checks() == []


@pytest.mark.parametrize("state", ["partial", "config_review_required", "interrupted"])
def test_source_safe_new_states_are_retained(database: Database, state: str) -> None:
    repo = repositories.SourceRepository(database)
    repo.record_status({"source_id": "computer", "state": state})
    assert repo.latest_statuses()[0]["state"] == state


def test_status_sanitizes_new_fields_on_write_and_read(database: Database) -> None:
    repo = repositories.SourceRepository(database)
    secret = "<html>contact@example.invalid title body</html>"
    repo.record_status({
        "source_id": "computer", "state": secret, "message": secret,
        "coverage_code": secret, "skip_reasons": {secret: 2, "queue_full": 1},
        "request_count": secret, "queued_count": -9,
    })
    status = repo.latest_statuses()[0]
    assert status["state"] == status["message"] == "error"
    assert status["coverage_code"] == ""
    assert status["request_count"] == status["queued_count"] == 0
    assert status["skip_reasons"] == {"other": 2, "queue_full": 1}
    with database.transaction() as connection:
        stored = dict(connection.execute("SELECT * FROM source_status").fetchone())
        assert secret not in json.dumps(stored)
        connection.execute("UPDATE source_status SET coverage_code = ?, message = ?, "
                           "skip_reasons_json = ?", (secret, secret, json.dumps({secret: 1})))
    assert secret not in json.dumps(repo.latest_statuses())


def test_queue_idempotence_preserves_first_seen_and_only_url_state(
    database: Database, sources: dict[str, SourceConfig],
) -> None:
    repo = queue(database, sources)
    assert repo.enqueue("computer", URL, now=NOW) is True
    assert repo.enqueue("computer", URL, now=NOW + timedelta(days=1)) is False
    assert repo.count("computer") == 1
    assert repo.oldest_ready("computer", now=NOW) == [{
        "source_id": "computer", "canonical_url": URL, "first_seen_at": NOW.isoformat(),
        "last_attempt_at": None, "transient_failures": 0, "paused_until": None,
    }]


def test_queue_capacity_is_per_source_and_duplicate_does_not_evict(
    database: Database, sources: dict[str, SourceConfig],
) -> None:
    repo = queue(database, sources)
    for number in range(200):
        assert repo.enqueue("computer", f"https://cs.shu.edu.cn/info/1/{number}.htm", now=NOW)
    assert not repo.enqueue("computer", URL, now=NOW)
    assert not repo.enqueue("computer", "https://cs.shu.edu.cn/info/1/extra.htm", now=NOW)
    assert repo.enqueue("graduate", "https://gs.shu.edu.cn/info/1/1.htm", now=NOW)
    assert repo.count("computer") == 200
    assert repo.count() == 201


def test_queue_expires_at_ninety_days_and_frees_capacity(
    database: Database, sources: dict[str, SourceConfig],
) -> None:
    repo = queue(database, sources)
    repo.enqueue("computer", URL, now=NOW)
    assert len(repo.oldest_ready("computer", now=NOW + timedelta(days=90, microseconds=-1))) == 1
    assert repo.oldest_ready("computer", now=NOW + timedelta(days=90)) == []
    assert repo.prune("computer", now=NOW + timedelta(days=90)) == 1
    assert repo.count() == 0
    assert repo.enqueue("computer", URL, now=NOW + timedelta(days=90))


def test_queue_pause_after_three_failures_and_retry_after_seven_days(
    database: Database, sources: dict[str, SourceConfig],
) -> None:
    repo = queue(database, sources)
    repo.enqueue("computer", URL, now=NOW)
    for number in range(3):
        repo.mark_transient_failure("computer", URL, now=NOW + timedelta(minutes=number))
        assert len(repo.oldest_ready("computer", now=NOW + timedelta(minutes=number))) == (
            1 if number < 2 else 0
        )
    ready_time = NOW + timedelta(minutes=2, days=7)
    assert repo.oldest_ready("computer", now=ready_time - timedelta(microseconds=1)) == []
    row = repo.oldest_ready("computer", now=ready_time)[0]
    assert row["transient_failures"] == 3
    assert row["last_attempt_at"] == "2026-10-06T00:02:00+00:00"
    repo.mark_transient_failure("computer", URL, now=ready_time)
    assert repo.oldest_ready("computer", now=ready_time) != []


@pytest.mark.parametrize("source_id,url", [
    ("unknown", URL), ("graduate", URL),
    ("computer", "http://cs.shu.edu.cn/info/1/1.htm"),
    ("computer", "https://cs.shu.edu.cn/private/1.htm"),
    ("computer", "https://cs.shu.edu.cn/info/1/file.pdf"),
    ("computer", "https://cs.shu.edu.cn:443/info/1/1.htm"),
    ("computer", "https://cs.shu.edu.cn/info/%252e%252e/1.htm"),
])
def test_queue_rejects_unsafe_url_without_persistence(
    database: Database, sources: dict[str, SourceConfig], source_id: str, url: str,
) -> None:
    repo = queue(database, sources)
    assert repo.enqueue(source_id, url, now=NOW) is False
    assert repo.count() == 0


def test_queue_read_revalidates_current_url_policy(
    database: Database, sources: dict[str, SourceConfig],
) -> None:
    repo = queue(database, sources)
    repo.enqueue("computer", URL, now=NOW)
    changed_sources = {**sources, "computer": replace(sources["computer"],
                                                     allowed_path_prefixes=("/xxgk/tzgg",))}
    assert queue(database, changed_sources).oldest_ready("computer", now=NOW) == []
    assert repo.count() == 1  # Reading alone does not mutate the queue.
    assert queue(database, changed_sources).prune("computer", now=NOW) == 1


def test_queue_oldest_order_limit_and_permanent_removal(
    database: Database, sources: dict[str, SourceConfig],
) -> None:
    repo = queue(database, sources)
    urls = ["https://cs.shu.edu.cn/info/1/b.htm", "https://cs.shu.edu.cn/info/1/a.htm"]
    for url in urls:
        repo.enqueue("computer", url, now=NOW)
    repo.enqueue("computer", URL, now=NOW + timedelta(minutes=1))
    assert [row["canonical_url"] for row in repo.oldest_ready("computer", limit=2, now=NOW)] == (
        list(reversed(urls))
    )
    assert repo.remove("computer", urls[1]) is True
    assert repo.remove("computer", urls[1]) is False
    assert repo.count("computer") == 2


def test_clear_queue_preserves_notices_and_personas(
    database: Database, sources: dict[str, SourceConfig],
) -> None:
    repo = queue(database, sources)
    repo.enqueue("computer", URL, now=NOW)
    repo.enqueue("graduate", "https://gs.shu.edu.cn/info/1/1.htm", now=NOW)
    repositories.NoticeRepository(database).upsert(NoticeDraft(
        "computer", URL, "课程安排通知", "课程安排公开说明",
    ))
    with database.transaction() as connection:
        connection.execute("INSERT INTO personas VALUES ('p', 'P', '{}', ?)", (NOW.isoformat(),))
    assert repo.clear("computer") == 1
    assert repo.count() == 1
    assert repo.clear() == 1
    assert repositories.NoticeRepository(database).count() == 1
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM personas").fetchone()[0] == 1


@pytest.mark.parametrize("operation", ["count", "clear", "prune"])
def test_empty_source_scope_never_becomes_all_sources(
    database: Database, sources: dict[str, SourceConfig], operation: str,
) -> None:
    repo = queue(database, sources)
    repo.enqueue("computer", URL, now=NOW)
    kwargs = {"now": NOW + timedelta(days=90)} if operation == "prune" else {}
    assert getattr(repo, operation)("", **kwargs) == 0
    assert repo.count("computer") == 1


def test_live_recheck_is_stable_and_excludes_snapshot_legacy_and_processed_urls(
    database: Database, sources: dict[str, SourceConfig],
) -> None:
    repo = repositories.NoticeRepository(database, OfficialNoticePolicy(sources))
    for suffix, source_id, status, verified_at in [
        ("b", "computer", ProofStatus.VERIFIED_LIVE, NOW),
        ("a", "computer", ProofStatus.VERIFIED_LIVE, NOW),
        ("old", "computer", ProofStatus.VERIFIED_LIVE, NOW - timedelta(days=1)),
        ("skip", "computer", ProofStatus.VERIFIED_LIVE, NOW - timedelta(days=2)),
        ("snapshot", "computer", ProofStatus.VERIFIED_SNAPSHOT, NOW - timedelta(days=3)),
        ("other", "graduate", ProofStatus.VERIFIED_LIVE, NOW - timedelta(days=3)),
    ]:
        host = "cs" if source_id == "computer" else "gs"
        repo.upsert(NoticeDraft(source_id, f"https://{host}.shu.edu.cn/info/1/{suffix}.htm",
                                "课程安排通知", "公开课程安排"),
                    verification=Verification(status, verified_at, "live:hash"))
    repo.upsert(NoticeDraft("computer", URL, "未核实通知", "未核实内容"))
    skip_url = "https://cs.shu.edu.cn/info/1/skip.htm"
    assert [row.canonical_url for row in repo.list_live_for_recheck("computer", {skip_url})] == [
        "https://cs.shu.edu.cn/info/1/old.htm", "https://cs.shu.edu.cn/info/1/a.htm",
        "https://cs.shu.edu.cn/info/1/b.htm",
    ]
