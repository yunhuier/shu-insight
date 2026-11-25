from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
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
from shu_insight.domain.models import FetchedPage, ParseState, ProofStatus, Verification
from shu_insight.ingest.http_client import HttpCache, PoliteHttpClient
from shu_insight.ingest.pipeline import IngestionPipeline
from shu_insight.services.notices import NoticeService
from shu_insight.sources.registry import load_source_registry

NOW = datetime(2026, 10, 6, tzinfo=UTC)
TITLE = "上海大学研究生课程办理通知"
BODY = (
    "研究生须按照培养计划办理课程申请，并由导师和学院审核。申请截止日期为2026年10月20日。"
    "相关学生须先完成个人培养计划，再通过研究生综合服务平台提交申请，"
    "由导师和学院审批后交研究生院审核。正式上课前请确认课程安排和办理要求。"
)
LIST = "https://gs.shu.edu.cn/xwlb/sy.htm"


def url(number):
    return f"https://gs.shu.edu.cn/info/1029/{number}.htm"


class LocalRun:
    """Real pipeline and HTTP budget over synthetic HTML and a temporary database."""

    def __init__(self, tmp_path, discoveries=()):
        self.database = Database(tmp_path / "queue.sqlite3")
        self.database.initialize()
        self.adapter = load_source_registry(Path("config/sources.toml"))["graduate"]
        self.adapter.config = replace(
            self.adapter.config, pagination_selector="a.next",
            pagination_url_pattern=r"/xwlb/sy(?:\d+)?\.htm",
            pagination_reviewed_on=date(2026, 10, 5),
            pagination_link_text=None,
        )
        self.notices = NoticeRepository(
            self.database, OfficialNoticePolicy({"graduate": self.adapter.config}),
        )
        self.sources = SourceRepository(self.database)
        self.queue = PendingCandidateRepository(self.database, {"graduate": self.adapter.config})
        self.pages = {LIST: list(discoveries)}
        self.links = {}
        self.failures = {}
        self.seen = []
        self.bodies = {}
        self.titles = {}
        self.dates = {}
        self.robots = "User-agent: *\nAllow: /"
        self.cache = None
        self.now = NOW

    def handler(self, request):
        target = str(request.url)
        self.seen.append(target)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=self.robots, headers={"content-type": "text/plain"})
        if target in self.failures:
            failure = self.failures[target]
            if failure == "network":
                raise httpx.ConnectError("private failure text", request=request)
            if failure == "redirect":
                return httpx.Response(302, headers={"location": target + "?detail=1"})
            if failure == "structure":
                return httpx.Response(200, text="<h1>通知</h1><main>changed</main>",
                                      headers={"content-type": "text/html"})
            if failure == "login":
                return httpx.Response(200, text="<title>用户登录</title>",
                                      headers={"content-type": "text/html"})
            return httpx.Response(failure, headers={"content-type": "text/html"})
        if target in self.pages:
            body = '<table width="755">' + ''.join(
                f'<tr id="line_u17_{index}"><td><a href="{item}">'
                f'{self.titles.get(item, TITLE)}</a></td>'
                f'<td>{self.dates.get(item, "")}</td></tr>'
                for index, item in enumerate(self.pages[target])
            ) + '</table>'
            if target in self.links:
                body += f'<a class="next" href="{self.links[target]}">下一页</a>'
        else:
            body = self.detail(self.bodies.get(target, BODY))
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})

    @staticmethod
    def detail(body=BODY):
        return f'<h1>{TITLE}</h1><div class="v_news_content">{body}</div>'

    def run(self):
        now = self.now

        class FixtureHttpClient(PoliteHttpClient):
            def fetch(self, *args, **kwargs):
                return replace(super().fetch(*args, **kwargs), fetched_at=now)

        with FixtureHttpClient(contact="offline@example.com", transport=httpx.MockTransport(
            self.handler), cache=self.cache, monotonic=lambda: 0.0, sleep=lambda _: None,
        ) as client:
            pipeline = IngestionPipeline(
                registry={"graduate": self.adapter}, client=client, notices=self.notices,
                sources=self.sources, now=lambda: now, today=lambda: now.date(),
            )
            run_id = ManualUpdateRunRepository(self.database).start(started_at=now)
            report = pipeline.run(run_id=run_id)
        return report.by_source["graduate"]

    def seed(self, target, *, at=NOW, proof=ProofStatus.VERIFIED_LIVE):
        # Public test text only; no live school verification is claimed by this fixture.
        draft = self.adapter.parse_detail(FetchedPage(
            "graduate", target, target, 200, "text/html", self.detail(), at,
        ))
        reference = "live:fixture" if proof is ProofStatus.VERIFIED_LIVE else "snapshot:fixture"
        self.notices.upsert(draft, verification=Verification(proof, at, reference),
                            summary_override=BODY)

    def details(self):
        return [target for target in self.seen if "/info/" in target]

    def seed_listing(self, target, *, published=""):
        """Prior parsed-list proof, rather than a title invented from a queued URL."""
        html = ('<table width="755"><tr id="line_u17_0"><td>'
                f'<a href="{target}">{TITLE}</a></td><td>{published}</td></tr></table>')
        discoveries, rejected = self.adapter.discover_with_rejections(html, LIST)
        assert not rejected and len(discoveries) == 1
        self.notices.upsert_listing(
            discoveries[0], list_url=LIST, checked_at=self.now - timedelta(days=1),
        )


def test_manual_click_does_not_fetch_or_promote_bare_url_only_pending(tmp_path):
    run = LocalRun(tmp_path, [url(500)])
    run.queue.enqueue("graduate", url(100), now=NOW - timedelta(days=1))

    result = run.run()

    assert run.details() == [url(500)]
    assert run.notices.find_by_url(url(100)) is None
    assert run.queue.urls("graduate") == {url(100)}
    assert result.queued_count == result.pending_count == 1
    assert [item.canonical_url for item in NoticeService(
        run.notices, authority_by_source_id={},
    ).all()] == [url(500)]


def test_inadmissible_persisted_listing_does_not_authorize_pending_detail(tmp_path):
    run = LocalRun(tmp_path, [url(500)])
    run.seed_listing(url(100))
    run.queue.enqueue("graduate", url(100), now=NOW)
    with run.database.transaction() as connection:
        connection.execute("UPDATE notices SET verification_ref = 'invalid-proof'")

    result = run.run()

    assert run.details() == [url(500)]
    assert result.queued_count == result.pending_count == 1
    assert {row.canonical_url for row in run.notices.list_browseable()} == {url(500)}


def test_persisted_listing_pending_keeps_priority_without_current_rediscovery(tmp_path):
    run = LocalRun(tmp_path, [url(500)])
    run.seed_listing(url(100))
    run.queue.enqueue("graduate", url(100), now=NOW - timedelta(days=1))
    run.seed(url(600), at=NOW - timedelta(days=2))
    run.failures[url(100)] = "network"
    assert run.run().pending_count == 1
    run.failures.clear()
    run.seen.clear()

    result = run.run()

    assert run.details() == [url(100), url(500)]
    assert result.pending_count == result.queued_count == result.listing_count == 0
    assert run.notices.find_by_url(url(100)).proof_status is ProofStatus.VERIFIED_LIVE


def test_later_page_failure_queues_only_prior_safe_list_evidence_without_details(tmp_path):
    run = LocalRun(tmp_path, [url(100), url(101)])
    second = "https://gs.shu.edu.cn/xwlb/sy2.htm"
    third = "https://gs.shu.edu.cn/xwlb/sy3.htm"
    run.links.update({LIST: second, second: third})
    run.pages[second] = [url(100)]
    run.failures[third] = "network"
    original = run.handler

    def handler(request):
        response = original(request)
        if str(request.url) == second:
            return httpx.Response(200, text=response.text.replace(TITLE, "[演示]虚构通知"),
                                  headers={"content-type": "text/html"})
        return response

    run.handler = handler
    result = run.run()

    assert result.state == "partial" and result.list_pages_read == 2
    assert run.details() == []
    assert run.queue.urls("graduate") == {url(101)}
    assert result.pending_count == 1 and result.queued_count == 0
    assert run.notices.find_by_url(url(100)) is None
    assert [item.canonical_url for item in run.notices.list_browseable()] == [url(101)]
    run.now += timedelta(hours=1)
    run.run()
    assert run.queue.oldest_ready("graduate", now=run.now)[0]["first_seen_at"] == NOW.isoformat()


def test_optional_due_history_rotates_without_partial_while_current_list_rechecks_each_time(
    tmp_path,
):
    run = LocalRun(tmp_path, [url(500)])
    run.seed(url(500))
    for n in range(100, 134):
        run.seed(url(n), at=NOW - timedelta(days=2))

    first = run.run()
    assert first.state == "success" and first.skipped_count == 0
    assert run.details() == [url(500)] + [url(n) for n in range(100, 131)]
    run.seen.clear()

    second = run.run()
    assert run.details() == [url(500), url(131), url(132), url(133)]
    assert second.state == "success"
    assert run.sources.latest_completed_checks()[0]["state"] == "success"
    run.seen.clear()
    run.now += timedelta(hours=23, minutes=59)
    assert run.run().state == "success"
    assert run.details() == [url(500)]
    run.seen.clear()
    run.now += timedelta(minutes=1)
    assert run.run().state == "success"
    assert run.details() == [url(500)] + [url(n) for n in range(100, 131)]


@pytest.mark.parametrize("published", [False, True])
def test_older_than_90_day_history_is_optional_and_not_requested(tmp_path, published):
    run = LocalRun(tmp_path, [url(500)])
    for n in range(100, 135):
        run.seed(url(n), at=NOW - timedelta(days=1 if published else 91))
    if published:
        with run.database.transaction() as connection:
            connection.execute("UPDATE notices SET published_at = ?", (
                (NOW - timedelta(days=91)).isoformat(),
            ))

    result = run.run()

    assert result.state == "success" and result.skipped_count == 0
    assert run.details() == [url(500)]


@pytest.mark.parametrize("proof,conditional,refreshed", [
    (ProofStatus.VERIFIED_LIVE, True, True),
    (ProofStatus.VERIFIED_LIVE, False, False),
    (ProofStatus.VERIFIED_SNAPSHOT, True, False),
])
def test_only_authorized_live_304_advances_historical_due_time(
    tmp_path, proof, conditional, refreshed,
):
    run = LocalRun(tmp_path, [url(500)])
    before = NOW - timedelta(days=2)
    run.seed(url(100), at=before, proof=proof)
    run.cache = HttpCache()
    if conditional:
        run.cache.save(url(100), etag='"fixture"', last_modified=None)
    run.failures[url(100)] = 304

    run.run()

    assert run.notices.find_by_url(url(100)).verified_at == (NOW if refreshed else before)
    run.seen.clear()
    run.run()
    assert (url(100) in run.details()) is not refreshed


def test_shared_32_candidates_prioritize_unparsed_then_new_then_all_parsed_proofs(tmp_path):
    run = LocalRun(tmp_path, [url(n) for n in range(200, 220)])
    for n in range(100, 110):
        run.seed_listing(url(n))
        run.queue.enqueue("graduate", url(n), now=NOW - timedelta(days=1))
    for n in range(300, 310):
        run.seed(url(n), proof=ProofStatus.VERIFIED_SNAPSHOT)
        run.pages[LIST].append(url(n))
    run.seed(url(400), at=NOW - timedelta(days=2))
    run.queue.enqueue("graduate", url(400), now=NOW - timedelta(days=3))
    result = run.run()
    assert run.details() == (
        [url(n) for n in range(100, 110)] + [url(n) for n in range(200, 220)]
        + [url(400), url(300)]
    )
    assert result.state == "partial" and result.message == "candidate_limit_reached"
    assert result.request_count == 34
    assert result.discovered_count == 30
    assert result.parsed_count == 41 and result.listing_count == 0
    assert result.queued_count == 0 and result.pending_count == 0


def test_fresh_transient_failure_is_queued_and_next_click_precedes_parsed_rechecks(tmp_path):
    run = LocalRun(tmp_path, [url(200)])
    run.seed(url(100), at=NOW - timedelta(days=2))
    run.failures[url(200)] = "network"
    first = run.run()
    row, = run.queue.oldest_ready("graduate", now=NOW)
    assert row["canonical_url"] == url(200) and row["transient_failures"] == 1
    assert row["first_seen_at"] == NOW.isoformat()
    assert first.queued_count == 0 and first.pending_count == 1
    assert first.listing_count == 1 and first.parsed_count == 1
    assert first.request_breakdown == {"robots": 1, "list": 1, "detail": 2, "retry": 1}
    run.failures.clear()
    run.seen.clear()
    second = run.run()
    assert run.details() == [url(200)]  # Historical 100 was just verified on the first click.
    assert second.pending_count == second.queued_count == second.listing_count == 0
    assert second.parsed_count == 2


def test_five_reviewed_pages_stop_before_actual_sixth_link_and_report_partial(tmp_path):
    run = LocalRun(tmp_path, [url(100)])
    pages = [LIST] + [f"https://gs.shu.edu.cn/xwlb/sy{n}.htm" for n in range(2, 7)]
    for index, page in enumerate(pages):
        run.pages[page] = [url(100 + index)]
        if index < 5:
            run.links[page] = pages[index + 1]
    result = run.run()
    assert result.list_pages_read == 5
    assert pages[5] not in run.seen
    assert result.state == "partial" and result.coverage_code == "page_limit_reached"
    assert result.request_breakdown == {"robots": 1, "list": 5, "detail": 5, "retry": 0}


def test_unreviewed_fourth_link_stops_at_last_reviewed_page(tmp_path):
    run = LocalRun(tmp_path, [url(100)])
    second = "https://gs.shu.edu.cn/xwlb/sy2.htm"
    third = "https://gs.shu.edu.cn/xwlb/sy3.htm"
    fourth = "https://gs.shu.edu.cn/xwlb/archive/4.htm"
    run.pages.update({second: [url(101)], third: [url(102)]})
    run.links.update({LIST: second, second: third, third: fourth})
    result = run.run()
    assert result.list_pages_read == 3 and fourth not in run.seen
    assert result.state == "partial"
    assert result.coverage_code in {"pagination_not_allowed", "pagination_failed"}


def test_candidate_quota_queues_unattempted_urls_and_next_click_advances(tmp_path):
    run = LocalRun(tmp_path, [url(n) for n in range(100, 134)])
    first = run.run()
    assert first.state == "partial"
    assert first.message == "candidate_limit_reached"
    assert first.skip_reasons == {"candidate_limit_reached": 2}
    assert first.request_count == 34
    assert first.queued_count == 0 and first.pending_count == 2
    assert first.listing_count == 2
    assert run.queue.urls("graduate") == {url(132), url(133)}
    assert run.notices.count() == 34
    assert len(run.notices.list_verified()) == 32
    run.seen.clear()
    second = run.run()
    assert run.details()[:2] == [url(132), url(133)]
    assert run.notices.count() == 34
    # Historical parsed rechecks also share the cap, so this click is partial.
    assert second.state == "partial"
    assert second.queued_count == 0
    assert second.pending_count == second.listing_count == 0
    assert len(run.notices.list_verified()) == 34
    assert run.sources.latest_completed_checks() == []
    with run.database.connect() as connection:
        assert [row[0] for row in connection.execute(
            "SELECT state FROM source_status ORDER BY id"
        )] == ["partial", "partial"]


def test_url_only_queue_is_invisible_until_official_list_evidence_exists(tmp_path):
    run = LocalRun(tmp_path)
    run.queue.enqueue("graduate", url(100), now=NOW)
    service = NoticeService(run.notices, authority_by_source_id={})

    assert service.all() == service.latest() == service.upcoming().all_notices == []
    assert run.notices.count() == 0


def test_official_listing_is_browseable_then_upgrades_with_same_identity(tmp_path):
    run = LocalRun(tmp_path, [url(n) for n in range(100, 134)])
    first = run.run()
    service = NoticeService(run.notices, authority_by_source_id={})

    assert first.queued_count == 0 and first.pending_count == 2
    listing = next(item for item in service.all() if item.canonical_url == url(133))
    assert listing.proof_status is ProofStatus.VERIFIED_LISTING
    assert listing.parse_state is ParseState.LISTING_ONLY
    assert listing.title == TITLE and listing.summary == ""
    assert listing.published_at is None and listing.deadline_at is None
    assert listing in service.latest() and listing in service.upcoming().unknown
    assert listing not in run.notices.list_verified()
    assert service.detail_evidence(listing.id) == {
        "duplicates": [], "possible_duplicates": [], "versions": [],
    }

    run.run()

    parsed = run.notices.find_by_url(url(133))
    assert parsed.id == listing.id and parsed.canonical_url == listing.canonical_url
    assert parsed.proof_status is ProofStatus.VERIFIED_LIVE
    assert parsed.parse_state is ParseState.PARSED and "课程申请" in parsed.summary
    assert len([item for item in service.all() if item.id == listing.id]) == 1


def test_later_attachment_withdraws_only_listing_and_does_not_reappear(tmp_path):
    run = LocalRun(tmp_path, [url(n) for n in range(100, 134)])
    run.run()
    listing = run.notices.find_by_url(url(133))
    parsed = run.notices.find_by_url(url(100))
    attachment = '<iframe src="/system/pdfjs/viewer.html"></iframe>'
    run.bodies[url(133)] = attachment
    run.bodies[url(100)] = attachment

    run.run()

    service = NoticeService(run.notices, authority_by_source_id={})
    assert listing is not None and listing.id not in {item.id for item in service.all()}
    assert run.notices.find_by_url(url(133)).proof_status is ProofStatus.ATTACHMENT_EXCLUDED
    assert run.notices.find_by_url(url(100)) == parsed
    run.run()
    assert listing.id not in {item.id for item in service.all()}


@pytest.mark.parametrize("failure", [404, "rejected_title", "robots"])
def test_later_permanent_rejection_hides_listing_without_downloading_attachment(tmp_path, failure):
    run = LocalRun(tmp_path, [url(n) for n in range(100, 134)])
    run.run()
    target = url(132)
    listing = run.notices.find_by_url(target)
    assert listing is not None
    if failure == "rejected_title":
        run.titles[target] = "[演示]虚构通知"
    elif failure == "robots":
        run.robots = "User-agent: *\nDisallow: /info/1029/132.htm\nAllow: /"
    else:
        run.failures[target] = failure

    run.run()

    assert target not in {item.canonical_url for item in run.notices.list_browseable()}
    assert run.notices.find_by_url(target).id == listing.id


def test_queued_omissions_advance_after_first_32_details_cannot_parse(tmp_path):
    run = LocalRun(tmp_path, [url(n) for n in range(100, 136)])
    for number in range(100, 132):
        run.failures[url(number)] = "structure"

    first = run.run()
    assert first.state == "partial"
    assert first.saved_count == 0
    assert first.request_count == 34
    assert first.queued_count == 0 and first.pending_count == 4
    assert run.queue.urls("graduate") == {url(n) for n in range(132, 136)}

    run.seen.clear()
    second = run.run()
    assert run.details()[:4] == [url(n) for n in range(132, 136)]
    assert second.saved_count == 4
    assert second.request_count == 34
    assert run.notices.count() == 36
    assert len(run.notices.list_verified()) == 4
    assert len(run.notices.list_browseable()) == 36


def test_public_notice_referring_to_future_name_list_is_not_discarded(tmp_path):
    run = LocalRun(tmp_path, [url(100)])
    run.bodies[url(100)] = (
        BODY + "录取名单将在结果确定后另行公示，本通知仅说明课程申请和审核办法。"
    )

    result = run.run()

    assert result.state == "success"
    assert result.saved_count == 1
    assert result.skip_reasons == {}
    assert run.notices.find_by_url(url(100)) is not None


def test_personal_award_details_are_removed_but_public_instructions_are_saved(tmp_path):
    run = LocalRun(tmp_path, [url(100)])
    run.bodies[url(100)] = (
        "经学院评审，现将2026年研究生国家奖学金评选结果予以公示：张三、李四。"
        "公示期为五个工作日，相关意见请向学院提交。请研究生院和各培养单位"
        "认真审核并及时反馈情况。"
    )

    result = run.run()

    assert result.saved_count == 1 and result.skip_reasons == {}
    saved = run.notices.find_by_url(url(100))
    assert saved is not None and "公示期为五个工作日" in saved.summary
    assert "张三" not in (tmp_path / "queue.sqlite3").read_bytes().decode("utf-8", "ignore")


def test_personal_admission_topic_is_read_for_safe_preview(tmp_path):
    run = LocalRun(tmp_path, [url(100)])
    run.titles[url(100)] = "拟录取人员名单（第一批）公示"

    result = run.run()

    assert result.saved_count == 1 and result.skip_reasons == {}
    assert run.details() == [url(100)]
    assert run.notices.find_by_url(url(100)).summary


@pytest.mark.parametrize(("title", "body"), [
    (
        "关于奖学金申请材料提交的通知",
        "经学校评审，张三、李四获得国家奖学金，获奖名单填报材料请于本周提交，"
        "各学院按规定办理申请和审核手续。请有关单位认真核对并反馈。",
    ),
    (
        "关于退学手续办理的通知",
        "张三同学作退学处理，退学手续办理流程由学院另行通知，请相关部门按规定"
        "提交材料并做好审核。有关单位应及时核对办理情况并反馈。",
    ),
    (
        "关于奖学金申请材料提交的通知",
        "张三、李四荣获国家奖学金并请各学院按奖学金申请材料流程办理后续手续。"
        "为确保申请审核工作顺利进行请及时向学校反馈。",
    ),
    (
        "关于处分申诉渠道及材料提交的通知",
        "张三同学被记过处分并可按处分申诉渠道提交材料办理后续手续。"
        "请相关学院做好工作安排并将申请材料及时送交主管部门。",
    ),
])
def test_result_clause_is_removed_while_other_instructions_persist(tmp_path, title, body):
    run = LocalRun(tmp_path, [url(100)])
    run.titles[url(100)] = title
    run.bodies[url(100)] = BODY + body

    result = run.run()

    assert result.saved_count == 1 and result.skip_reasons == {}
    saved = run.notices.find_by_url(url(100))
    assert saved is not None and "课程申请" in saved.summary
    assert "张三" not in (tmp_path / "queue.sqlite3").read_bytes().decode("utf-8", "ignore")


def test_ready_queue_precedes_new_discoveries_without_old_bucket_reservations(tmp_path):
    run = LocalRun(tmp_path, [url(n) for n in range(200, 230)])
    for n in range(100, 106):
        run.seed_listing(url(n))
        run.queue.enqueue("graduate", url(n), now=NOW - timedelta(days=1))
    result = run.run()
    assert run.details() == [url(n) for n in range(100, 106)] + [url(n) for n in range(200, 226)]
    assert result.queued_count == 0 and result.pending_count == 4
    assert run.queue.urls("graduate") == {url(n) for n in range(226, 230)}
    assert result.state == "partial"
    assert result.request_count == 34
    run.pages[LIST] = [url(200)]
    run.seen.clear()
    result = run.run()
    assert run.details()[:4] == [url(n) for n in range(226, 230)]
    assert result.queued_count == 0


@pytest.mark.parametrize("old_count,new_count", [(0, 32), (30, 2), (32, 0)])
def test_shared_slots_allow_either_unparsed_bucket_to_fill_candidate_cap(
    tmp_path, old_count, new_count,
):
    run = LocalRun(tmp_path, [url(n) for n in range(200, 200 + new_count)] or [url(900)])
    if not new_count:
        run.seed(url(900))
    for n in range(100, 100 + old_count):
        run.seed_listing(url(n))
        run.queue.enqueue("graduate", url(n), now=NOW - timedelta(days=1))
    run.run()
    assert len([target for target in run.details() if target != url(900)]) == 32
    if old_count == 30:
        assert run.details() == (
            [url(n) for n in range(100, 130)] + [url(200), url(201)]
        )
    assert run.queue.count("graduate") == 0


def test_physical_budget_retains_interrupted_and_unattempted_safe_cross_page_urls(tmp_path):
    run = LocalRun(tmp_path, [url(n) for n in range(100, 108)])
    second_list = "https://gs.shu.edu.cn/xwlb/sy2.htm"
    run.links[LIST] = second_list
    run.pages[second_list] = [url(n) for n in range(106, 126)]
    for n in range(100, 126):
        run.failures[url(n)] = "redirect"
    result = run.run()
    assert result.state == "partial"
    assert result.message == "request_budget_exhausted"
    assert result.request_count == 40
    assert result.list_pages_read == 2
    assert result.saved_count == 18
    # Attempt 40 starts 118; its redirect and all following details wait for a click.
    assert run.details()[-1] == url(118)
    assert result.queued_count == 0 and result.pending_count == 8
    assert run.queue.urls("graduate") == {url(n) for n in range(118, 126)}
    listing = run.notices.find_by_url(url(119))
    assert listing.proof_status is ProofStatus.VERIFIED_LISTING and listing.summary == ""
    assert len(run.notices.list_verified()) == 18
    assert len(run.notices.list_browseable()) == 26
    assert len(run.seen) == 40
    status, = run.sources.latest_statuses()
    assert status["queued_count"] == 0 and status["pending_count"] == 8
    assert status["request_breakdown"] == {"robots": 1, "list": 2, "detail": 37, "retry": 0}


def test_parsed_rechecks_include_snapshots_in_oldest_first_order_and_detect_version(tmp_path):
    run = LocalRun(tmp_path, [url(500)])
    run.seed(url(500), proof=ProofStatus.VERIFIED_SNAPSHOT)
    for n, days in [(102, 1), (101, 2), (100, 2)]:
        run.seed(url(n), at=NOW - timedelta(days=days))
    run.bodies[url(100)] = BODY + "办理时间调整为2026年10月22日。"
    first = run.run()
    assert run.details() == [url(500), url(100), url(101), url(102)]
    assert first.updated_count >= 1
    assert run.notices.find_by_url(url(500)).proof_status is ProofStatus.VERIFIED_LIVE
    changed = run.notices.find_by_url(url(100))
    assert "办理时间调整" in changed.summary
    assert len(run.notices.detail_evidence(changed.id)["versions"]) == 2
    run.seen.clear()
    second = run.run()
    assert second.updated_count == 0
    assert len(run.notices.detail_evidence(changed.id)["versions"]) == 2


def test_snapshot_without_obtained_body_never_promotes(tmp_path):
    run = LocalRun(tmp_path, [url(500)])
    run.seed(url(500), proof=ProofStatus.VERIFIED_SNAPSHOT)
    run.failures[url(500)] = "network"
    result = run.run()
    assert result.state == "partial"
    assert result.message == "transport_detail_connect"
    assert result.queued_count == 0
    assert result.skip_reasons == {"temporary_failure": 1}
    assert run.notices.find_by_url(url(500)).proof_status is ProofStatus.VERIFIED_SNAPSHOT


@pytest.mark.parametrize("stage", ["robots", "list", "detail"])
@pytest.mark.parametrize(("error_type", "kind", "retries", "failure_state"), [
    (httpx.RemoteProtocolError, "remote_protocol", 0, "connection_failed"),
    (httpx.LocalProtocolError, "local_protocol", 0, "connection_failed"),
    (httpx.WriteError, "write", 0, "connection_failed"),
    (httpx.WriteTimeout, "write_timeout", 1, "connection_timeout"),
    (httpx.PoolTimeout, "pool_timeout", 1, "connection_timeout"),
    (httpx.CloseError, "close", 0, "connection_failed"),
    (httpx.UnsupportedProtocol, "unsupported_protocol", 0, "connection_failed"),
    (httpx.TransportError, "other", 0, "connection_failed"),
    (httpx.NetworkError, "other", 0, "connection_failed"),
    (httpx.ProtocolError, "other", 0, "connection_failed"),
    (httpx.TimeoutException, "other", 1, "connection_failed"),
])
def test_transport_subclass_boundary_preserves_safe_status_without_promoting_unread_body(
    tmp_path, stage, error_type, kind, retries, failure_state, caplog,
):
    run = LocalRun(tmp_path, [url(500)])
    target = {"robots": "https://gs.shu.edu.cn/robots.txt", "list": LIST,
              "detail": url(500)}[stage]
    secret = "synthetic-transport-secret private@example.invalid http://user:password@proxy.invalid"
    original = run.handler

    def handler(request):
        if str(request.url) == target:
            run.seen.append(target)
            raise error_type(secret, request=request)
        return original(request)

    run.handler = handler
    result = run.run()
    state = "partial" if stage == "detail" else failure_state
    code = f"transport_{stage}_{kind}"
    assert (result.state, result.message) == (state, code)
    status, = run.sources.latest_statuses()
    assert (status["state"], status["message"]) == (state, code)
    breakdown = {"robots": 1, "list": int(stage != "robots"),
                 "detail": int(stage == "detail"), "retry": retries if stage != "robots" else 0}
    assert result.request_breakdown == status["request_breakdown"] == breakdown
    assert result.request_count == status["request_count"] == sum(breakdown.values())
    assert run.seen.count(target) == (1 if stage == "robots" else 1 + retries)
    assert result.skip_reasons == ({"temporary_failure": 1} if stage == "detail" else {})
    assert run.notices.list_verified() == []
    if stage == "detail":
        record = run.notices.find_by_url(url(500))
        assert record.proof_status is ProofStatus.VERIFIED_LISTING and record.summary == ""
        assert run.queue.count("graduate") == 1
    else:
        assert run.notices.list_browseable() == [] and run.details() == []
    with run.database.connect() as connection:
        dump = "\n".join(connection.iterdump())
        assert connection.execute("SELECT state, message FROM source_status").fetchone()[:] == (
            state, code,
        )
    assert "synthetic-transport-secret" not in dump + repr(result) + repr(status) + caplog.text
    assert b"synthetic-transport-secret" not in run.database.path.read_bytes()


@pytest.mark.parametrize("verified", [False, True])
def test_undated_live_body_reuses_only_admissible_prior_publication(tmp_path, verified):
    run = LocalRun(tmp_path, [url(500)])
    draft = run.adapter.parse_detail(FetchedPage(
        "graduate", url(500), url(500), 200, "text/html", run.detail(), NOW,
    ))
    prior_date = datetime(2099, 1, 1, tzinfo=UTC)
    proof = Verification(ProofStatus.VERIFIED_LIVE, NOW, "live:fixture") if verified else None
    run.notices.upsert(replace(draft, published_at=prior_date), verification=proof)
    assert run.run().state == "success"
    record = run.notices.list_verified()[0]
    assert record.proof_status is ProofStatus.VERIFIED_LIVE
    assert record.published_at == (prior_date if verified else None)


def test_pending_forced_refresh_server_failure_remains_transient(tmp_path):
    run = LocalRun(tmp_path, [url(500)])
    run.seed_listing(url(100))
    run.queue.enqueue("graduate", url(100), now=NOW)
    run.cache = HttpCache()
    run.cache.save(url(100), etag='"fixture"', last_modified=None)
    original = run.handler

    def handler(request):
        if str(request.url) == url(100):
            run.seen.append(str(request.url))
            return httpx.Response(304 if "if-none-match" in request.headers else 503,
                                  headers={"content-type": "text/html"})
        return original(request)

    run.handler = handler
    result = run.run()
    assert result.state == "partial"
    assert result.queued_count == 0 and result.pending_count == 1
    assert result.skip_reasons["temporary_failure"] == 1
    assert run.queue.oldest_ready("graduate", now=NOW)[0]["transient_failures"] == 1
    assert run.notices.find_by_url(url(100)).proof_status is ProofStatus.VERIFIED_LISTING
    assert run.details() == [url(100), url(100), url(500)]


@pytest.mark.parametrize("bucket", ["ready", "paused", "live", "live_304"])
def test_fresh_synthetic_list_title_blocks_every_bucket_and_removes_pending(tmp_path, bucket):
    run = LocalRun(tmp_path, [url(100), url(500)])
    run.titles[url(100)] = "[演示]虚构通知"
    before = None
    if bucket.startswith("live"):
        run.seed(url(100), at=NOW - timedelta(days=1))
        before = run.notices.find_by_url(url(100)).verified_at
    else:
        run.queue.enqueue("graduate", url(100), now=NOW)
        if bucket == "paused":
            for _ in range(3):
                run.queue.mark_transient_failure("graduate", url(100), now=NOW)
    if bucket == "live_304":
        run.cache = HttpCache()
        run.cache.save(url(100), etag='"fixture"', last_modified=None)
        run.failures[url(100)] = 304
    result = run.run()
    assert url(100) not in run.details()
    assert run.queue.urls("graduate") == set()
    assert result.queued_count == 0
    assert result.state == "success"
    assert result.skip_reasons["sensitive_title"] == 1
    if before is not None:
        assert run.notices.find_by_url(url(100)).verified_at == before
    else:
        assert run.notices.find_by_url(url(100)) is None


def test_rejected_only_reviewed_list_removes_paused_backlog_without_detail(tmp_path):
    run = LocalRun(tmp_path, [url(100)])
    run.titles[url(100)] = "[演示]虚构通知"
    run.queue.enqueue("graduate", url(100), now=NOW)
    for _ in range(3):
        run.queue.mark_transient_failure("graduate", url(100), now=NOW)
    result = run.run()
    assert result.state == "success"
    assert result.list_pages_read == 1
    assert result.queued_count == 0
    assert result.skip_reasons == {"sensitive_title": 1}
    assert run.details() == []


def test_later_page_rejection_overrides_earlier_allowed_queue_discovery(tmp_path):
    run = LocalRun(tmp_path, [url(100), url(500)])
    second_list = "https://gs.shu.edu.cn/xwlb/sy2.htm"
    run.links[LIST] = second_list
    run.pages[second_list] = [url(100)]
    run.queue.enqueue("graduate", url(100), now=NOW)
    original = run.handler

    def handler(request):
        response = original(request)
        if str(request.url) == second_list:
            return httpx.Response(200, text=response.text.replace(TITLE, "[演示]虚构通知"),
                                  headers={"content-type": "text/html"})
        return response

    run.handler = handler
    result = run.run()
    assert result.list_pages_read == 2
    assert url(100) not in run.details()
    assert result.queued_count == 0
    assert run.notices.find_by_url(url(100)) is None
    assert url(100) not in {item.canonical_url for item in run.notices.list_browseable()}


def test_matching_queued_discovery_keeps_current_publication_hint_for_undated_body(tmp_path):
    run = LocalRun(tmp_path, [url(100)])
    run.dates[url(100)] = "2026/09/12"
    run.queue.enqueue("graduate", url(100), now=NOW)
    result = run.run()
    assert result.state == "success"
    assert result.queued_count == 0
    assert run.notices.find_by_url(url(100)).published_at.isoformat() == "2026-09-11T16:00:00+00:00"


@pytest.mark.parametrize("rediscovered", [False, True])
def test_queued_listing_upgrade_keeps_prior_date_without_new_date(tmp_path, rediscovered):
    run = LocalRun(tmp_path, [url(100)] if rediscovered else [url(500)])
    run.seed_listing(url(100), published="2026/09/12")
    before = run.notices.find_by_url(url(100))
    assert before.published_at == datetime(2026, 9, 11, 16, tzinfo=UTC)
    run.queue.enqueue("graduate", url(100), now=NOW)

    result = run.run()

    after = run.notices.find_by_url(url(100))
    assert result.state == "success"
    assert run.details() == ([url(100)] if rediscovered else [url(100), url(500)])
    assert after.proof_status is ProofStatus.VERIFIED_LIVE
    assert after.id == before.id and run.notices.count() == (1 if rediscovered else 2)
    assert after.published_at == datetime(2026, 9, 11, 16, tzinfo=UTC)
    assert result.pending_count == result.listing_count == 0
    with run.database.connect() as connection:
        versions = connection.execute(
            "SELECT payload_json FROM notice_versions WHERE notice_id = ? ORDER BY rowid",
            (after.id,),
        ).fetchall()
    assert len(versions) == 2
    assert all(json.loads(row[0])["published_at"] == "2026-09-11T16:00:00+00:00"
               for row in versions)


@pytest.mark.parametrize("detail_date,current_date,expected", [
    ("", "2026/09/13", "2026-09-12T16:00:00+00:00"),
    ("2026-09-14", "2026/09/13", "2026-09-13T16:00:00+00:00"),
    ("2026-09-14", "", "2026-09-13T16:00:00+00:00"),
])
def test_queued_listing_upgrade_prefers_detail_then_current_list_date(
    tmp_path, detail_date, current_date, expected,
):
    run = LocalRun(tmp_path, [url(100)])
    run.seed_listing(url(100), published="2026/09/12")
    run.queue.enqueue("graduate", url(100), now=NOW)
    run.dates[url(100)] = current_date
    original = run.handler

    def handler(request):
        response = original(request)
        if str(request.url) == url(100) and detail_date:
            return httpx.Response(
                200, text=response.text + f'<time datetime="{detail_date}"></time>',
                headers={"content-type": "text/html"},
            )
        return response

    run.handler = handler
    assert run.run().state == "success"
    assert run.notices.find_by_url(url(100)).published_at.isoformat() == expected


@pytest.mark.parametrize("prior_state", ["legacy", "withdrawn", "invalid_proof"])
def test_queued_listing_upgrade_does_not_reuse_inadmissible_prior_date(tmp_path, prior_state):
    run = LocalRun(tmp_path, [url(100)])
    run.seed_listing(url(100), published="2026/09/12")
    if prior_state == "withdrawn":
        run.notices.withdraw_listing(url(100))
    else:
        with run.database.transaction() as connection:
            if prior_state == "legacy":
                connection.execute("UPDATE notices SET proof_status = 'unverified_legacy'")
            else:
                connection.execute("UPDATE notices SET verification_ref = 'invalid-proof'")
    run.queue.enqueue("graduate", url(100), now=NOW)

    assert run.run().state == "success"

    record = run.notices.find_by_url(url(100))
    assert record.proof_status is ProofStatus.VERIFIED_LIVE and record.published_at is None


def test_expiry_pruned_before_backlog_and_verified_history_preserved(tmp_path):
    run = LocalRun(tmp_path, [url(500)])
    run.queue.enqueue("graduate", url(100), now=NOW - timedelta(days=90))
    run.seed(url(100), at=NOW - timedelta(days=100))
    result = run.run()
    assert result.queued_count == 0
    assert result.skip_reasons["queue_removed"] == 1
    assert run.notices.find_by_url(url(100)) is not None
    assert result.state == "success"


@pytest.mark.parametrize("failure", [404, "attachment", "synthetic_title"])
def test_permanently_rejected_list_proven_pending_withdraws_browseable_card(tmp_path, failure):
    run = LocalRun(tmp_path, [url(500)])
    run.seed_listing(url(100))
    run.queue.enqueue("graduate", url(100), now=NOW)
    if failure == "attachment":
        run.bodies[url(100)] = '<iframe src="/__local/notice.pdf"></iframe>'
    elif failure == "synthetic_title":
        original = run.handler

        def handler(request):
            response = original(request)
            if str(request.url) == url(100):
                return httpx.Response(
                    200, text=response.text.replace(TITLE, "[演示]虚构通知"),
                    headers={"content-type": "text/html"},
                )
            return response

        run.handler = handler
    else:
        run.failures[url(100)] = failure
    result = run.run()
    assert result.queued_count == 0
    assert run.notices.find_by_url(url(100)).proof_status is ProofStatus.ATTACHMENT_EXCLUDED
    assert url(100) not in {row.canonical_url for row in run.notices.list_browseable()}
    assert not any(".pdf" in target for target in run.seen)
    assert {404: "not_found", "attachment": "attachment_blocked",
            "synthetic_title": "sensitive_title"}[failure] in result.skip_reasons


@pytest.mark.parametrize("failure", ["network", 503])
def test_transient_failures_pause_and_rediscovery_cannot_bypass_pause(tmp_path, failure):
    run = LocalRun(tmp_path, [url(100), url(500)])
    run.queue.enqueue("graduate", url(100), now=NOW)
    run.failures[url(100)] = failure
    for _ in range(3):
        result = run.run()
        assert result.state == "partial"
        assert result.queued_count == 0 and result.pending_count == 1
        assert result.listing_count == 1
    row = run.queue.oldest_ready("graduate", limit=200, now=NOW)
    assert row == []
    run.seen.clear()
    result = run.run()
    assert url(100) not in run.details()
    assert result.state == "partial"
    assert result.queued_count == 0 and result.pending_count == 1
    assert result.skip_reasons["queue_paused"] == 1
    assert "private failure text" not in str(run.sources.latest_statuses())


def test_stored_queue_read_does_not_prune_but_button_run_revalidates_urls(tmp_path):
    run = LocalRun(tmp_path, [url(500)])
    run.queue.enqueue("graduate", url(100), now=NOW)
    run.adapter.config = replace(
        run.adapter.config, allowed_path_prefixes=("/xwlb/sy", "/info/1029/500.htm"),
    )
    assert run.queue.urls("graduate") == {url(100)}
    assert run.queue.count("graduate") == 1
    result = run.run()
    assert result.queued_count == 0
    assert result.skip_reasons == {"queue_removed": 1}
    assert url(100) not in run.details()


def test_robots_login_response_halts_source_without_touching_pending(tmp_path):
    run = LocalRun(tmp_path, [url(500)])
    run.queue.enqueue("graduate", url(100), now=NOW)
    run.robots = "<title>用户登录</title>"
    result = run.run()
    assert result.state == "robots_html_or_login"
    assert result.queued_count == 1
    assert run.seen == ["https://gs.shu.edu.cn/robots.txt"]


def test_optional_parsed_rechecks_use_shared_cap_without_marking_required_scope_partial(tmp_path):
    run = LocalRun(tmp_path, [url(n) for n in range(100, 112)])
    for n in range(200, 225):
        run.seed(url(n), at=NOW - timedelta(days=1))
    result = run.run()
    assert result.request_count == 34
    assert result.state == "success"
    assert result.message == "" and result.skipped_count == 0
    assert result.coverage_code == "reviewed_range_complete"
    assert run.details() == [url(n) for n in range(100, 112)] + [url(n) for n in range(200, 220)]
    assert result.queued_count == 0
    assert url(220) not in run.details()


def test_parsed_recheck_redirect_budget_stop_reports_partial(tmp_path):
    run = LocalRun(tmp_path, [url(n) for n in range(100, 113)])
    for n in range(200, 220):
        run.seed(url(n), at=NOW - timedelta(days=1))
    for n in range(200, 220):
        run.failures[url(n)] = "redirect"
    before = run.notices.find_by_url(url(212)).verified_at
    result = run.run()
    assert result.request_count == 40
    assert result.state == "partial"
    assert result.coverage_code == "request_budget_exhausted"
    assert result.queued_count == 0
    assert run.notices.find_by_url(url(212)).verified_at == before
    assert result.message == "request_budget_exhausted"
    assert url(212) in run.details()
    assert url(212) + "?detail=1" not in run.details()
    assert url(213) not in run.details()


def test_physical_budget_failure_preserves_other_sources_independent_allowance(tmp_path):
    run = LocalRun(tmp_path, [url(n) for n in range(100, 126)])
    for n in range(100, 126):
        run.failures[url(n)] = "redirect"
    undergraduate = load_source_registry(Path("config/sources.toml"))["undergraduate"]
    undergraduate_url = "https://bksy.shu.edu.cn/info/1015/321714.htm"
    original = run.handler

    def handler(request):
        if request.url.host == "gs.shu.edu.cn":
            return original(request)
        run.seen.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /",
                                  headers={"content-type": "text/plain"})
        body = (f'<div id="body"><div class="only-listPage"><div class="only-list"><ul>'
                f'<li><a href="{undergraduate_url}">{TITLE}</a><span></span></li>'
                '</ul></div></div></div>'
                if str(request.url) == undergraduate.config.list_url else run.detail())
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})

    run.notices.policy = OfficialNoticePolicy({
        "graduate": run.adapter.config, "undergraduate": undergraduate.config,
    })
    with PoliteHttpClient(contact="offline@example.com", transport=httpx.MockTransport(handler),
                          monotonic=lambda: 0.0, sleep=lambda _: None) as client:
        report = IngestionPipeline(
            registry={"graduate": run.adapter, "undergraduate": undergraduate}, client=client,
            notices=run.notices, sources=run.sources, now=lambda: NOW, today=lambda: NOW.date(),
        ).run()
    assert report.by_source["graduate"].request_count == 40
    assert report.by_source["graduate"].message == "request_budget_exhausted"
    assert report.by_source["undergraduate"].request_count == 3
    assert report.by_source["undergraduate"].state == "success"
    assert report.by_source["undergraduate"].saved_count == 1


@pytest.mark.parametrize("failure,reason", [
    (403, "forbidden"), (429, "rate_limited"), ("login", "login_or_captcha"),
])
def test_pending_hard_stop_retains_queue_and_stops_new_details(tmp_path, failure, reason):
    run = LocalRun(tmp_path, [url(500)])
    run.seed_listing(url(100))
    run.queue.enqueue("graduate", url(100), now=NOW)
    run.failures[url(100)] = failure
    result = run.run()
    assert result.state == reason
    assert result.queued_count == 1
    assert run.details() == [url(100)]
    assert run.notices.list_verified() == []
    assert [item.canonical_url for item in run.notices.list_browseable()] == [url(500)]
    assert run.notices.find_by_url(url(100)).proof_status is ProofStatus.ATTACHMENT_EXCLUDED


def test_pending_detail_structure_failure_removes_bad_row_and_checks_next_notice(tmp_path):
    run = LocalRun(tmp_path, [url(500)])
    run.seed_listing(url(100))
    run.queue.enqueue("graduate", url(100), now=NOW)
    run.failures[url(100)] = "structure"
    result = run.run()
    assert result.state == "partial"
    assert result.message == "structure_changed"
    assert result.queued_count == 0
    assert result.skip_reasons == {"structure_changed": 1}
    assert run.details() == [url(100), url(500)]
    assert run.notices.find_by_url(url(100)).proof_status is ProofStatus.VERIFIED_LISTING
    assert run.notices.find_by_url(url(100)).summary == ""
    assert run.notices.find_by_url(url(500)) is not None


def test_pending_robots_denial_removes_only_rejected_candidate_and_next_click_advances(tmp_path):
    run = LocalRun(tmp_path, [url(500)])
    run.seed_listing(url(100))
    run.seed_listing(url(101))
    run.queue.enqueue("graduate", url(100), now=NOW)
    run.queue.enqueue("graduate", url(101), now=NOW)
    run.robots = "User-agent: *\nDisallow: /info/1029/100.htm\nAllow: /"
    result = run.run()
    assert result.state == "robots_disallowed"
    assert result.queued_count == 0 and result.pending_count == 1
    assert run.queue.urls("graduate") == {url(101)}
    assert result.skip_reasons == {"robots_disallowed": 1}
    assert run.details() == []
    assert run.notices.list_verified() == []
    assert {item.canonical_url for item in run.notices.list_browseable()} == {url(101), url(500)}
    assert run.notices.find_by_url(url(100)).proof_status is ProofStatus.ATTACHMENT_EXCLUDED
    assert run.sources.latest_completed_checks() == []
    run.seen.clear()
    second = run.run()
    assert second.state == "success"
    assert second.queued_count == 0
    assert run.details() == [url(101), url(500)]
    assert run.notices.count() == 3  # Includes the withdrawn listing tombstone for 100.


def test_queue_capacity_refusal_keeps_url_only_candidates_invisible(tmp_path):
    run = LocalRun(tmp_path, [url(n) for n in range(300, 320)])
    for n in range(100, 300):
        run.queue.enqueue("graduate", url(n), now=NOW)
    for n in range(300, 320):
        run.failures[url(n)] = "redirect"
    result = run.run()
    assert result.request_count == 40
    assert result.queued_count == result.pending_count == 200
    assert result.skip_reasons["queue_full"] > 0
    listing = run.notices.find_by_url(url(319))
    assert listing.proof_status is ProofStatus.VERIFIED_LISTING and listing.summary == ""
    assert run.notices.find_by_url(url(299)) is None
    assert not set(run.details()).intersection(url(n) for n in range(100, 300))
    with run.database.connect() as connection:
        assert len(connection.execute("SELECT * FROM pending_candidates").fetchone().keys()) == 6
