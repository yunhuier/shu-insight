from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import (
    ManualUpdateRunRepository,
    NoticeRepository,
    SourceRepository,
)
from shu_insight.db.schema import Database
from shu_insight.domain.models import (
    DiscoveredNotice,
    FetchedPage,
    NoticeDraft,
    ParseState,
    ProofStatus,
    SourceConfig,
    Verification,
)
from shu_insight.ingest.http_client import (
    PoliteHttpClient,
    RequestBudgetExceeded,
    RobotsUnavailable,
)
from shu_insight.ingest.pipeline import IngestionPipeline, UpdateEvent
from shu_insight.services.update import UpdateService
from shu_insight.sources.registry import load_source_registry

FIXTURE_DIR = Path("tests/fixtures/sites")
SAFE_DETAIL_URL = "https://gs.shu.edu.cn/info/1029/173242.htm"
UNDERGRAD_DETAIL_URL = "https://bksy.shu.edu.cn/info/1015/321714.htm"
SENSITIVE_DETAIL_URL = "https://gs.shu.edu.cn/info/3000/3002.htm"


def graduate_listing(links: list[tuple[str, str]], *, published: str = "") -> str:
    rows = ''.join(
        f'<tr id="line_u17_{index}"><td><a href="{url}">{title}</a></td>'
        f'<td>{published}</td></tr>'
        for index, (url, title) in enumerate(links)
    )
    return f'<table width="755">{rows}</table>'


class FixtureClient:
    def __init__(self) -> None:
        self.statuses: dict[str, int] = {}
        self.list_from_cache: set[str] = set()
        self.not_modified_details: set[str] = set()
        self.requested_urls: list[str] = []

    def set_status(self, source_id: str, status: int) -> None:
        self.statuses[source_id] = status

    def fetch(
        self,
        source: SourceConfig,
        url: str,
        *,
        force_refresh: bool = False,
        purpose: str = "detail",
        redirect_target_guard=None,
    ) -> FetchedPage:
        del force_refresh, redirect_target_guard
        self.requested_urls.append(url)
        is_list = url == source.list_url
        body = ""
        status = (
            304
            if not is_list and url in self.not_modified_details
            else self.statuses.get(source.id, 200)
        )
        if is_list:
            if status == 200 and source.id == "graduate":
                body = graduate_listing([
                    (SAFE_DETAIL_URL, OFFICIAL_TITLE),
                    (SENSITIVE_DETAIL_URL, '授予硕士学位人员公告'),
                ], published="2026/09/11")
            elif status == 200 and source.id == "undergraduate":
                body = ('<div id="body"><div class="only-listPage">'
                        '<div class="only-list"><ul><li>'
                        f'<a href="{UNDERGRAD_DETAIL_URL}">'
                        '上海大学2026级本科新生秋季学期选课通知</a>'
                        '<span>2026-08-28</span></li></ul></div></div></div>')
        elif status == 200:
            if url == SENSITIVE_DETAIL_URL:
                body = ('<h1>授予硕士学位人员公告</h1><div class="v_news_content">'
                        '<iframe src="/__local/result.pdf"></iframe></div>')
            elif source.id == "graduate":
                body = (f'<h1>{OFFICIAL_TITLE}</h1><div class="v_news_content">'
                        f'{OFFICIAL_BODY}</div>')
            else:
                body = ('<h1>上海大学2026级本科新生秋季学期选课通知</h1>'
                        '<div class="v_news_content">'
                        '2026级本科新生线上选课第一轮为2026年9月3日9时至17时30分，'
                        '第二轮为9月6日13时至17时30分。学生须按培养方案、课程代码、'
                        '开课校区和先修要求核对课程，正式上课前确认课表。'
                        '</div>')
        return FetchedPage(
            source_id=source.id,
            url=url,
            final_url=url,
            status_code=status,
            content_type="text/html; charset=utf-8",
            body=body,
            fetched_at=datetime(2026, 9, 21, tzinfo=UTC),
            from_cache=(is_list and source.id in self.list_from_cache) or status == 304,
        )


@pytest.fixture
def database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "pipeline.sqlite3")
    database.initialize()
    return database


@pytest.fixture
def client() -> FixtureClient:
    return FixtureClient()


@pytest.fixture
def pipeline(database: Database, client: FixtureClient) -> IngestionPipeline:
    all_sources = load_source_registry(Path("config/sources.toml"))
    selected = {
        source_id: all_sources[source_id]
        for source_id in ("undergraduate", "graduate")
    }
    return IngestionPipeline(
        registry=selected,
        client=client,
        today=lambda: date(2026, 10, 6),
        notices=NoticeRepository(database, OfficialNoticePolicy(
            {key: adapter.config for key, adapter in selected.items()})),
        sources=SourceRepository(database),
    )


def test_one_source_429_does_not_stop_other_sources(
    pipeline: IngestionPipeline, client: FixtureClient
) -> None:
    client.set_status("undergraduate", 429)

    report = pipeline.run()

    assert report.by_source["undergraduate"].state == "rate_limited"
    assert report.by_source["graduate"].saved_count == 1


def test_unrecognized_list_structure_is_reported_and_persisted_as_failure(
    pipeline: IngestionPipeline, client: FixtureClient, database: Database,
) -> None:
    class ChangedListClient(FixtureClient):
        def fetch(
            self, source, url, *, force_refresh=False, redirect_target_guard=None, purpose="detail",
        ):
            page = super().fetch(source, url, force_refresh=force_refresh)
            if url != source.list_url:
                return page
            return FetchedPage(
                page.source_id, page.url, page.final_url, page.status_code,
                page.content_type,
                '<main><section class="renamed-list"><a href="/info/1029/173242.htm">'
                f'{OFFICIAL_TITLE}</a></section></main>',
                page.fetched_at,
            )

    pipeline.client = ChangedListClient()
    report = pipeline.run(["graduate"])

    assert report.by_source["graduate"].state == "structure_changed"
    assert report.by_source["graduate"].saved_count == 0
    diagnostic = report.by_source["graduate"].listing_diagnostic
    assert diagnostic is not None
    assert diagnostic.link_count == 1
    assert diagnostic.scoped_article_link_count == 1
    assert diagnostic.selected_anchor_count == 0
    assert diagnostic.html_characters > 0
    assert any(event.kind == "source_failed" and event.message == "structure_changed"
               for event in report.events)
    status = SourceRepository(Database(database.path)).latest_statuses()[0]
    assert status["state"] == "structure_changed"
    assert "listing_diagnostic" not in status


def test_listing_diagnostic_groups_only_safe_structure_not_notice_content(pipeline):
    from shu_insight.ingest.pipeline import _diagnose_listing

    adapter = pipeline.registry["graduate"]
    html = (
        '<html><body><header><a href="/info/9000/1.htm">页眉通知</a></header>'
        '<section class="public-list secret@example.invalid"><table class="list">'
        '<tr class="news-item"><td>'
        '<a href="../../info/1029/173242.htm">不应显示的真实标题</a></td>'
        '<td>2026/10/16</td></tr>'
        '</table></section><footer><a href="/info/9000/2.htm">页脚通知</a>'
        '</footer></body></html>'
    )

    diagnostic = _diagnose_listing(adapter, html, adapter.config.list_url)

    assert diagnostic.scoped_article_link_count == 3
    assert diagnostic.selected_anchor_count == 0
    assert sum(item.count for item in diagnostic.link_patterns) == 3
    assert any("public-list" in item.ancestor_signature for item in diagnostic.link_patterns)
    assert any(item.nearby_date for item in diagnostic.link_patterns)
    assert all(item.raw_info_marker for item in diagnostic.link_patterns)
    rendered = str(diagnostic)
    assert "不应显示的真实标题" not in rendered
    assert "secret@example.invalid" not in rendered
    assert "173242" not in rendered


def test_listing_diagnostic_distinguishes_resolved_info_from_raw_href(pipeline):
    from shu_insight.ingest.pipeline import _diagnose_listing

    adapter = pipeline.registry["graduate"]
    diagnostic = _diagnose_listing(
        adapter,
        '<ul class="news"><li><a href="info/1029/173242.htm">公告</a></li></ul>',
        'https://gs.shu.edu.cn/',
    )

    assert diagnostic.scoped_article_link_count == 1
    assert diagnostic.link_patterns[0].raw_info_marker is False


def test_topic_discovery_is_fetched_but_attachment_body_stays_excluded(
    pipeline: IngestionPipeline,
    client: FixtureClient,
    database: Database,
) -> None:
    report = pipeline.run(["graduate"])

    assert report.by_source["graduate"].saved_count == 1
    assert SENSITIVE_DETAIL_URL in client.requested_urls
    assert NoticeRepository(database).find_by_url(SENSITIVE_DETAIL_URL) is None
    saved = NoticeRepository(database).find_by_url(SAFE_DETAIL_URL)
    assert saved is not None
    assert saved.deadline_at is not None
    assert "研究生" in saved.title
    assert saved.proof_status is ProofStatus.VERIFIED_LIVE


def test_cached_unchanged_list_still_rechecks_existing_live_detail(
    pipeline: IngestionPipeline,
    client: FixtureClient,
) -> None:
    first = pipeline.run(["graduate"])
    original = pipeline.notices.find_by_url(SAFE_DETAIL_URL)
    client.list_from_cache.add("graduate")
    second = pipeline.run(["graduate"])
    third = pipeline.run(["graduate"])

    assert first.by_source["graduate"].saved_count == 1
    assert second.by_source["graduate"].saved_count == 0
    assert second.by_source["graduate"].skipped_count == 2
    assert second.by_source["graduate"].skip_reasons == {
        "attachment_blocked": 1, "content_unchanged": 1,
    }
    assert third.by_source["graduate"].saved_count == 0
    assert client.requested_urls.count(SAFE_DETAIL_URL) == 3
    assert pipeline.notices.find_by_url(SAFE_DETAIL_URL).published_at == original.published_at
    assert len(pipeline.notices.detail_evidence(original.id)["versions"]) == 1


def test_live_recheck_uses_current_list_publication_hint_without_date_drift(pipeline, client):
    pipeline.run(["graduate"])
    original = pipeline.notices.find_by_url(SAFE_DETAIL_URL)

    class UpdatedPublication(FixtureClient):
        def fetch(
            self, source, url, *, force_refresh=False, redirect_target_guard=None, purpose="detail",
        ):
            page = super().fetch(source, url, force_refresh=force_refresh,
                                 redirect_target_guard=redirect_target_guard)
            return replace(page, body=page.body.replace("2026/09/11", "2026/09/12"))

    pipeline.client = UpdatedPublication()
    report = pipeline.run(["graduate"])
    changed = pipeline.notices.find_by_url(SAFE_DETAIL_URL)
    assert changed.published_at.isoformat() == "2026-09-11T16:00:00+00:00"
    assert changed.published_at != original.published_at
    assert report.by_source["graduate"].updated_count == 1
    assert len(pipeline.notices.detail_evidence(original.id)["versions"]) == 2


def test_independent_update_services_skip_bodyless_304_for_existing_detail(
    database: Database,
) -> None:
    registry = load_source_registry(Path("config/sources.toml"))
    selected = {"undergraduate": registry["undergraduate"]}

    class UndergraduateAttachments(FixtureClient):
        attachment_url = 'https://bksy.shu.edu.cn/info/1015/900002.htm'

        def fetch(self, source, url, **kwargs):
            page = super().fetch(source, url, **kwargs)
            if url == source.list_url and page.status_code == 200:
                body = page.body.replace('</ul>', '<li><a href="' + self.attachment_url
                                         + '">附件说明通知</a></li></ul>')
                return replace(page, body=body)
            if url == self.attachment_url:
                return replace(page, body='<h1>附件说明通知</h1><div class="v_news_content">'
                               '<iframe src="/__local/result.pdf"></iframe></div>')
            return page

    first_client = UndergraduateAttachments()
    first = UpdateService(
        database=database, registry=selected, client_factory=lambda: nullcontext(first_client),
    ).update_all()

    second_client = UndergraduateAttachments()
    second_client.not_modified_details.add(UNDERGRAD_DETAIL_URL)
    second = UpdateService(
        database=database, registry=selected, client_factory=lambda: nullcontext(second_client),
    ).update_all()

    assert first.by_source["undergraduate"].saved_count == 1
    assert second.by_source["undergraduate"].saved_count == 0
    assert second.by_source["undergraduate"].skipped_count == 2
    assert second.by_source["undergraduate"].skip_reasons == {
        "attachment_blocked": 1, "unchanged_detail": 1,
    }
    assert second_client.requested_urls.count(UNDERGRAD_DETAIL_URL) == 1


def test_update_service_emits_source_and_notice_progress(
    pipeline: IngestionPipeline,
    client: FixtureClient,
) -> None:
    events: list[UpdateEvent] = []
    client.set_status("shu_main", 429)
    registry = load_source_registry(Path('config/sources.toml'))

    report = UpdateService(
        database=pipeline.notices.database,
        registry={key: registry[key] for key in ('shu_main', 'undergraduate')},
        client_factory=lambda: nullcontext(client), taxonomy=pipeline.taxonomy,
        today=pipeline.today, now=pipeline.now,
    ).update_all(events.append)

    assert report.by_source["undergraduate"].saved_count == 1
    assert {event.kind for event in events} >= {
        "source_started",
        "source_failed",
        "notice_saved",
        "source_finished",
    }


def test_source_batches_duplicate_mining_refresh_once(
    pipeline: IngestionPipeline,
    monkeypatch,
) -> None:
    calls = 0
    original = NoticeRepository.refresh_mining

    def counted(repository):
        nonlocal calls
        calls += 1
        return original(repository)

    monkeypatch.setattr(NoticeRepository, "refresh_mining", counted)

    report = pipeline.run(["graduate"])

    assert report.saved_count == 1
    assert calls == 1


def test_partial_source_failure_still_refreshes_saved_notice_mining(
    database: Database,
    monkeypatch,
) -> None:
    second_url = "https://gs.shu.edu.cn/info/1029/173243.htm"

    class PartialFailureClient(FixtureClient):
        def fetch(
            self, source, url, *, force_refresh=False, redirect_target_guard=None, purpose="detail",
        ):
            if url == second_url:
                return FetchedPage(
                    source.id, url, url, 429, "text/html", "",
                    datetime(2026, 9, 21, tzinfo=UTC),
                )
            page = super().fetch(source, url, force_refresh=force_refresh)
            if url == source.list_url:
                extra = (
                    '<tr id="line_u17_2"><td><a href="/info/1029/173243.htm">'
                    "上海大学研究生课程办理通知</a></td><td></td></tr>"
                )
                return FetchedPage(
                    page.source_id, page.url, page.final_url,
                    page.status_code, page.content_type,
                    page.body.replace("</table>", extra + "</table>"),
                    page.fetched_at,
                )
            return page

    calls = 0
    original = NoticeRepository.refresh_mining

    def counted(repository):
        nonlocal calls
        calls += 1
        return original(repository)

    monkeypatch.setattr(NoticeRepository, "refresh_mining", counted)
    registry = load_source_registry(Path("config/sources.toml"))
    report = IngestionPipeline(
        registry={"graduate": registry["graduate"]},
        client=PartialFailureClient(),
        today=lambda: date(2026, 10, 6),
        notices=NoticeRepository(database, OfficialNoticePolicy(
            {"graduate": registry["graduate"].config})),
        sources=SourceRepository(database),
    ).run()

    assert report.by_source["graduate"].state == "rate_limited"
    assert report.by_source["graduate"].saved_count == 1
    assert NoticeRepository(database).count() == 1
    assert calls == 1


OFFICIAL_URL = "https://gs.shu.edu.cn/info/1029/173242.htm"
OFFICIAL_TITLE = "上海大学2026级硕博连读研究生、直博生 申请课程免修及成绩认定通知"
OFFICIAL_BODY = (
    "2026级硕博连读研究生、直博生符合条件者可申请相应课程免修或已修课程的学分、成绩认定。"
    "申请时间为2026年9月11日至10月11日，申请截止日期为2026年10月11日。"
    "相关学生须先完成个人培养计划，再通过研究生综合服务平台提交申请，"
    "由导师和学院审批后交研究生院审核。"
)


class OfficialFixtureClient:
    def __init__(self, *, body=OFFICIAL_BODY, container=True, modified=False):
        self.body = body
        self.container = container
        self.modified = modified
        self.force_refreshes = 0
        self.fetched_at = datetime(2026, 9, 25, tzinfo=UTC)
        self.not_modified_final_url = None

    def fetch(
        self, source, url, *, force_refresh=False, redirect_target_guard=None, purpose="detail",
    ):
        if force_refresh:
            self.force_refreshes += 1
        if url == source.list_url:
            html = graduate_listing([(OFFICIAL_URL, OFFICIAL_TITLE)], published="2026/09/11")
            status = 200
        elif self.modified and not force_refresh:
            html = ""
            status = 304
        else:
            opening = '<div class="v_news_content">' if self.container else '<section>'
            closing = '</div>' if self.container else '</section>'
            html = (f'<h1>{OFFICIAL_TITLE}</h1>{opening}{self.body}{closing}'
                    '<footer>info@shu.edu.cn <a href="/login">登录</a></footer>')
            status = 200
        final_url = (
            self.not_modified_final_url
            if status == 304 and self.not_modified_final_url else url
        )
        return FetchedPage(source.id, url, final_url, status, "text/html", html,
                           self.fetched_at, from_cache=status == 304)


def official_pipeline(database, client):
    registry = load_source_registry(Path("config/sources.toml"))
    sources = {"graduate": registry["graduate"]}
    return IngestionPipeline(
        registry=sources, client=client,
        notices=NoticeRepository(database, OfficialNoticePolicy(
            {key: adapter.config for key, adapter in sources.items()})),
        sources=SourceRepository(database),
        today=lambda: date(2026, 10, 6),
    )


def test_footer_email_and_login_link_do_not_block_official_article(database):
    report = official_pipeline(database, OfficialFixtureClient()).run()
    saved = NoticeRepository(database).find_by_url(OFFICIAL_URL)
    assert report.saved_count == 1
    assert saved is not None and saved.proof_status is ProofStatus.VERIFIED_LIVE
    assert "info@shu.edu.cn" not in saved.summary
    assert "课程免修" in saved.summary


@pytest.mark.parametrize("body", [
    OFFICIAL_BODY + "学号：2312345678。",
    OFFICIAL_BODY + "联系人：13800138000，学号：2312345678。",
])
def test_identifiers_are_redacted_without_discarding_public_article(database, body):
    report = official_pipeline(database, OfficialFixtureClient(body=body)).run()
    assert report.saved_count == 1
    saved = NoticeRepository(database).find_by_url(OFFICIAL_URL)
    assert saved is not None and "课程免修" in saved.summary
    assert b"2312345678" not in database.path.read_bytes()


def test_all_personal_body_is_parsed_without_enrichment_or_mining(database, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("empty safe text must not be enriched or mined")

    monkeypatch.setattr("shu_insight.ingest.pipeline.extract_fields", forbidden)
    monkeypatch.setattr("shu_insight.ingest.pipeline.classify_notice", forbidden)
    monkeypatch.setattr(NoticeRepository, "refresh_mining", forbidden)
    client = OfficialFixtureClient(body="<p>张三同学获得国家奖学金。</p>")
    report = official_pipeline(database, client).run()
    saved = NoticeRepository(database).find_by_url(OFFICIAL_URL)
    assert report.saved_count == 1 and report.by_source["graduate"].state == "success"
    assert saved is not None and saved.parse_state is ParseState.PARSED
    assert saved.proof_status is ProofStatus.VERIFIED_LIVE and saved.summary == ""
    assert not saved.categories and not saved.education_levels and saved.deadline_at is None
    assert "张三".encode() not in database.path.read_bytes()


def test_two_all_redacted_articles_have_distinct_response_proof(database):
    second = "https://gs.shu.edu.cn/info/1029/173243.htm"

    class PersonalClient(OfficialFixtureClient):
        def fetch(self, source, url, **kwargs):
            page = super().fetch(source, url, **kwargs)
            if url == source.list_url:
                return replace(page, body=graduate_listing([
                    (OFFICIAL_URL, "奖学金结果公示"), (second, "录取结果公告"),
                ]))
            name = "张三" if url == OFFICIAL_URL else "李四"
            return replace(page, body=f"<h1>结果公告</h1><div class='v_news_content'>"
                                      f"<p>{name}同学获得国家奖学金。</p></div>")

    pipeline = official_pipeline(database, PersonalClient())
    report = pipeline.run()
    rows = pipeline.notices.list_verified()
    assert report.saved_count == 2 and len(rows) == 2
    assert all(row.summary == "" and row.verification_ref.startswith("live:") for row in rows)
    assert len({row.verification_ref for row in rows}) == 2


def test_detail_title_identifiers_are_redacted_before_persistence(database):
    class IdentifierTitleClient(OfficialFixtureClient):
        def fetch(self, source, url, **kwargs):
            page = super().fetch(source, url, **kwargs)
            if url != source.list_url:
                return replace(page, body=page.body.replace(
                    OFFICIAL_TITLE, "奖学金申请咨询13812345678 学号2312345678",
                ))
            return page

    assert official_pipeline(database, IdentifierTitleClient()).run().saved_count == 1
    saved = NoticeRepository(database).find_by_url(OFFICIAL_URL)
    assert saved is not None and "奖学金申请" in saved.title
    for raw in ("13812345678", "2312345678"):
        assert raw.encode() not in database.path.read_bytes()


@pytest.mark.parametrize("listing_only", [False, True])
@pytest.mark.parametrize("title,public_fragment,private_names", [
    ("关于张三同学记过处分的公告", "记过处分", ("张三",)),
    ("研究生张三同学获得国家奖学金", "国家奖学金", ("张三",)),
    ("关于给予李四同学退学处理的公告", "退学处理", ("李四",)),
    ("关于张三、李四同学通过复试的公告", "通过复试", ("张三", "李四")),
    ("关于张三获得国家奖学金的公示", "国家奖学金", ("张三",)),
    ("关于张成果同学获得国家奖学金的公示", "国家奖学金", ("张成果",)),
    ("关于张成果获得国家奖学金的公示", "国家奖学金", ("张成果",)),
    ("关于张成果获得科研奖励的通知", "科研奖励", ("张成果",)),
    ("关于赵平台通过评审的公示", "通过评审", ("赵平台",)),
    ("关于欧阳娜娜获得科研奖励的通知", "科研奖励", ("欧阳娜娜",)),
    ("关于朵云同学获得科研奖励的通知", "科研奖励", ("朵云",)),
    ("关于研究生朵云获得科研奖励的通知", "科研奖励", ("朵云",)),
    ("关于研究生张成果获得科研奖励的通知", "科研奖励", ("张成果",)),
])
def test_named_person_title_is_redacted_before_notice_and_version_persistence(
    database, listing_only, title, public_fragment, private_names,
):
    class NamedTitleClient(OfficialFixtureClient):
        def fetch(self, source, url, **kwargs):
            if listing_only and url != source.list_url:
                raise RequestBudgetExceeded()
            page = super().fetch(source, url, **kwargs)
            return replace(page, body=page.body.replace(OFFICIAL_TITLE, title))

    pipeline = official_pipeline(database, NamedTitleClient())
    result = pipeline.run().by_source["graduate"]

    saved = pipeline.notices.list_browseable()[0]
    expected_proof = ProofStatus.VERIFIED_LISTING if listing_only else ProofStatus.VERIFIED_LIVE
    assert saved.proof_status is expected_proof
    assert result.listing_count == int(listing_only)
    assert public_fragment in saved.title and "[姓名已隐藏]" in saved.title
    if listing_only:
        assert saved.summary == ""
    else:
        assert "课程免修" in saved.summary
    with database.connect() as connection:
        title_row = connection.execute("SELECT title FROM notices").fetchone()[0]
        versions = connection.execute("SELECT payload_json FROM notice_versions").fetchall()
    assert versions
    for payload in (title_row, *(row[0] for row in versions)):
        assert public_fragment in payload
        for name in private_names:
            assert name not in payload


@pytest.mark.parametrize("listing_only", [False, True])
@pytest.mark.parametrize("title", [
    "关于开展研究生国家奖学金评选工作的通知",
    "关于学生纪律处分管理办法的通知",
    "关于研究生同学获奖条件的说明",
    "各位同学通过学校平台提交申请材料的通知",
    "关于学校获得国家级奖励的通知",
    "关于上海大学获得科研奖励的通知",
    "关于科研项目通过评审的公示",
    "关于研究课题通过考核的公示",
    "关于科研团队获得科研奖励的通知",
    "关于实验中心获得科研奖励的通知",
    "关于研究所获得科研奖励的通知",
    "关于科研成果获得科研奖励的通知",
    "关于服务平台通过评审的公示",
    "关于软件系统通过评审的公示",
    "关于设计方案获得科研奖励的通知",
    "关于申请人通过复试后提交材料的通知",
    "关于研究生课程通过评审的公示",
    "关于我校同学通过学校平台提交申请材料的通知",
    "关于本院同学通过学校平台提交申请材料的通知",
    "关于该校同学通过学校平台提交申请材料的通知",
    "关于各院同学通过学校平台提交申请材料的通知",
])
def test_public_title_is_preserved_in_notice_and_version(database, listing_only, title):
    class PublicTitleClient(OfficialFixtureClient):
        def fetch(self, source, url, **kwargs):
            if listing_only and url != source.list_url:
                raise RequestBudgetExceeded()
            page = super().fetch(source, url, **kwargs)
            return replace(page, body=page.body.replace(OFFICIAL_TITLE, title))

    pipeline = official_pipeline(database, PublicTitleClient())
    pipeline.run()

    saved = pipeline.notices.list_browseable()[0]
    expected_proof = ProofStatus.VERIFIED_LISTING if listing_only else ProofStatus.VERIFIED_LIVE
    assert saved.proof_status is expected_proof and saved.title == title
    with database.connect() as connection:
        title_row = connection.execute("SELECT title FROM notices").fetchone()[0]
        versions = connection.execute("SELECT payload_json FROM notice_versions").fetchall()
    assert title_row == title
    assert versions and all(json.loads(row[0])["title"] == title for row in versions)


@pytest.mark.parametrize("private_html", [
    "<p>学号：</p><p>2312345678</p>",
    "<p>张三同学通过复试。</p>",
    "<p>研究生张三同学获得国家奖学金。</p>",
    "<p>优秀本科生李四同学通过复试。</p>",
    "<p>张三荣获一等奖。</p>",
    "<p>获奖名单如下：</p><span>张三</span>",
    "<table><tr><td>申请人</td><td>等级</td></tr>"
    "<tr><td>张三</td><td>一等奖</td></tr></table>",
])
def test_split_private_markup_never_reaches_notices_or_versions(database, private_html):
    class FragmentedClient(OfficialFixtureClient):
        def fetch(self, source, url, **kwargs):
            page = super().fetch(source, url, **kwargs)
            if url != source.list_url:
                return replace(page, body=page.body.replace(
                    OFFICIAL_TITLE,
                    "咨询电话 <span>138</span> <span>1234</span> <span>5678</span> 通知",
                ))
            return page

    pipeline = official_pipeline(database, FragmentedClient(body=OFFICIAL_BODY + private_html))
    assert pipeline.run().saved_count == 1
    saved = pipeline.notices.find_by_url(OFFICIAL_URL)
    assert saved is not None and "课程免修" in saved.summary
    with database.connect() as connection:
        notice_payload = str([tuple(row) for row in connection.execute("SELECT * FROM notices")])
        versions = str([tuple(row) for row in connection.execute("SELECT * FROM notice_versions")])
    for payload in (notice_payload, versions):
        for private in ("2312345678", "张三", "李四", "一等奖", "138 1234 5678"):
            assert private not in payload
    assert all(fragment not in saved.title for fragment in ("138", "1234", "5678"))


@pytest.mark.parametrize("body", [
    "申请人通过录取系统确认入学信息，请按通知要求提交材料。",
    "获奖者通过学校平台提交材料，相关通知请以官网为准。",
    "同学们通过学校平台提交申请材料，具体时间安排请查看官网。",
    "全体同学们通过学校平台提交申请材料，具体时间安排请查看官网。",
    "研究生同学们通过学校平台提交申请材料，具体时间安排请查看官网。",
    "同学们可以通过学校平台提交申请材料，具体时间安排请查看官网。",
    "各位同学请通过学校平台提交申请材料，具体时间安排请查看官网。",
    "申请人需要通过录取系统确认入学信息，请核对通知要求。",
    "全体研究生应当通过学校官网办理申请手续，具体时间请查看通知。",
    "请申请人通过学校平台提交学位授予申请材料，具体时间安排请查看官网。",
    "申请人请通过学校平台查询奖学金获奖条件，并按通知要求提交材料。",
])
def test_generic_role_instructions_are_persisted_as_read_body(database, body):
    pipeline = official_pipeline(database, OfficialFixtureClient(body=body))
    report = pipeline.run()
    saved = pipeline.notices.find_by_url(OFFICIAL_URL)
    assert report.saved_count == 1
    assert saved is not None and saved.parse_state is ParseState.PARSED
    assert saved.summary == body
    versions = pipeline.notices.detail_evidence(saved.id)["versions"]
    assert versions[0]["summary"] == body


@pytest.mark.parametrize("private_html", [
    "<p>2026级研究生张三获得国家奖学金。</p>",
    "<p>2027届A班Alice荣获一等奖。</p>",
    "<p>研究生张三</p><p>通过复试。</p>",
    "<p>优秀本科生李四</p><p>通过复试。</p>",
    "<p>2026级计算机专业研究生王小明</p><p>荣获国家奖学金。</p>",
    "<p>A班Alice</p><p>通过面试。</p>",
    "<p>2026级申请人张三</p><p>通过复试。</p>",
    "<p>2026级研究生张三通过复试后可通过学校平台确认入学。</p>",
    "<p>研究生张三获得国家奖学金后请通过学校系统提交材料。</p>",
    "<p>同学们通过学校平台提交材料时获知2026级研究生张三获得国家奖学金。</p>",
    "<p>同学们通过学校平台查询到张三被录取为硕士研究生。</p>",
    "<p>同学们通过学校系统查询到王小明当选学生会主席。</p>",
    "<p>同学们可以通过学校官网核对李四荣获优秀学生称号的结果。</p>",
    "<p>同学们通过学校官网办理材料并查看李四被评为优秀毕业生的结果。</p>",
    "<p>张三当选学生会主席后同学们可以通过学校平台提交意见。</p>",
    "<p>申请人获得国家奖学金后须通过学校平台查询张三被录取为硕士研究生的结果。</p>",
    "<p>通过学校系统查询张三被授予硕士学位的信息。</p>",
    "<p>同学们通过学校平台查询到2026级A班Alice入选校级优秀学生名单。</p>",
    "<p>申请人通过学校平台提交学位授予申请材料时获知张三被授予硕士学位。</p>",
    "<p>申请人请通过学校平台查询奖学金获奖条件时获知李四已获奖。</p>",
    "<p>学校决定授予申请人张三硕士学位。</p>",
    "<p>同学们通过学校平台查询到学校授予申请人张三硕士学位。</p>",
])
@pytest.mark.parametrize("mixed", [False, True])
def test_subject_shape_cannot_leak_results_into_notice_or_version(database, private_html, mixed):
    safe = "请各学院按通知要求提交申请材料并核对截止时间。" if mixed else ""
    pipeline = official_pipeline(database, OfficialFixtureClient(
        body=private_html + (f"<p>{safe}</p>" if safe else ""),
    ))
    assert pipeline.run().saved_count == 1
    saved = pipeline.notices.find_by_url(OFFICIAL_URL)
    assert saved is not None and saved.parse_state is ParseState.PARSED
    assert saved.proof_status is ProofStatus.VERIFIED_LIVE and saved.summary == safe
    assert pipeline.notices.detail_evidence(saved.id)["versions"][0]["summary"] == safe
    with database.connect() as connection:
        for table in ("notices", "notice_versions"):
            payload = str([tuple(row) for row in connection.execute(f"SELECT * FROM {table}")])
            for private in (
                "张三", "李四", "王小明", "Alice", "奖学金", "一等奖", "通过复试", "通过面试",
            ):
                assert private not in payload


@pytest.mark.parametrize(("prefix", "tail"), [
    ("申请人获得国家奖学金后", "须通过学校平台提交申请材料，具体时间安排请查看官网。"),
    ("候选人当选学生会主席后", "需通过学校官网办理任职手续，具体时间安排请查看官网。"),
    ("学生荣获优秀学生称号后", "请通过学校平台提交申请材料，具体时间安排请查看官网。"),
    ("申请人被录取为硕士研究生后", "应当通过学校系统确认入学信息，具体时间安排请查看官网。"),
])
def test_only_safe_procedural_tail_reaches_notice_and_version(database, prefix, tail):
    pipeline = official_pipeline(database, OfficialFixtureClient(body=prefix + tail))
    assert pipeline.run().saved_count == 1
    saved = pipeline.notices.find_by_url(OFFICIAL_URL)
    assert saved is not None and saved.parse_state is ParseState.PARSED
    assert saved.proof_status is ProofStatus.VERIFIED_LIVE and saved.summary == tail
    assert pipeline.notices.detail_evidence(saved.id)["versions"][0]["summary"] == tail
    with database.connect() as connection:
        for table in ("notices", "notice_versions"):
            payload = str([tuple(row) for row in connection.execute(f"SELECT * FROM {table}")])
            assert prefix not in payload


@pytest.mark.parametrize("contact", ["13800138000", "course-contact@example.com"])
def test_public_article_contacts_are_redacted_without_losing_action_or_deadline(database, contact):
    report = official_pipeline(database, OfficialFixtureClient(
        body=f"联系人：{contact}。" + OFFICIAL_BODY,
    )).run()
    saved = NoticeRepository(database).find_by_url(OFFICIAL_URL)
    assert report.saved_count == 1
    assert saved is not None and "申请" in saved.summary
    assert saved.deadline_at == datetime(2026, 10, 10, 16, tzinfo=UTC)  # Shanghai 10/11 00:00
    assert contact.encode() not in database.path.read_bytes()


def test_public_contact_in_title_is_redacted_before_persistence(database):
    class ContactTitleClient(OfficialFixtureClient):
        def fetch(self, source, url, **kwargs):
            page = super().fetch(source, url, **kwargs)
            return replace(page, body=page.body.replace(
                OFFICIAL_TITLE, OFFICIAL_TITLE + " 联系邮箱 course-contact@example.com",
            ))

    assert official_pipeline(database, ContactTitleClient()).run().saved_count == 1
    saved = NoticeRepository(database).find_by_url(OFFICIAL_URL)
    assert saved is not None and "课程免修" in saved.title
    assert b"course-contact@example.com" not in database.path.read_bytes()


def test_detail_budget_exhaustion_keeps_redacted_official_listing_without_summary(database):
    class ExhaustedDetailClient(OfficialFixtureClient):
        def fetch(self, source, url, **kwargs):
            if url != source.list_url:
                raise RequestBudgetExceeded()
            page = super().fetch(source, url, **kwargs)
            return replace(page, body=graduate_listing([(
                OFFICIAL_URL, OFFICIAL_TITLE + " 联系电话 13800138000 学号：2312345678",
            )]))

    pipeline = official_pipeline(database, ExhaustedDetailClient())
    result = pipeline.run().by_source["graduate"]

    assert result.message == "request_budget_exhausted"
    listing = pipeline.notices.list_browseable()[0]
    assert listing.proof_status is ProofStatus.VERIFIED_LISTING
    assert listing.summary == "" and listing.published_at is None
    assert "13800138000" not in listing.title and "2312345678" not in listing.title
    assert b"13800138000" not in database.path.read_bytes()
    assert b"2312345678" not in database.path.read_bytes()


def test_detail_redirect_upgrades_the_discovered_url_without_duplicate_identity(database):
    class RedirectedDetailClient(OfficialFixtureClient):
        def fetch(self, source, url, **kwargs):
            page = super().fetch(source, url, **kwargs)
            return replace(page, final_url=url + "?detail=1") if url == OFFICIAL_URL else page

    pipeline = official_pipeline(database, RedirectedDetailClient())
    pipeline.run()

    assert pipeline.notices.find_by_url(OFFICIAL_URL).proof_status is ProofStatus.VERIFIED_LIVE
    assert pipeline.notices.find_by_url(OFFICIAL_URL + "?detail=1") is None


def test_undated_redirected_live_recheck_preserves_publication_and_version(database):
    class RedirectedRecheckClient(OfficialFixtureClient):
        redirect = False

        def fetch(self, source, url, **kwargs):
            page = super().fetch(source, url, **kwargs)
            if not self.redirect:
                return page
            if url == source.list_url:
                return replace(page, body=graduate_listing([(OFFICIAL_URL, OFFICIAL_TITLE)]))
            return replace(page, final_url=url + "?detail=1")

    client = RedirectedRecheckClient()
    pipeline = official_pipeline(database, client)
    pipeline.run()
    before = pipeline.notices.find_by_url(OFFICIAL_URL)
    client.redirect = True

    result = pipeline.run().by_source["graduate"]

    after = pipeline.notices.find_by_url(OFFICIAL_URL)
    assert after.published_at == datetime(2026, 9, 10, 16, tzinfo=UTC)
    assert after.id == before.id and after.content_fingerprint == before.content_fingerprint
    assert result.saved_count == result.updated_count == 0
    assert len(pipeline.notices.detail_evidence_verified(after.id)["versions"]) == 1
    assert pipeline.notices.find_by_url(OFFICIAL_URL + "?detail=1") is None


@pytest.mark.parametrize("existing_kind", ["fresh", "listing", "parsed"])
@pytest.mark.parametrize("force_refresh", [False, True])
@pytest.mark.parametrize("target", [
    "https://gs.shu.edu.cn/info/1029/173242.pdf",
    "https://example.invalid/info/1029/173242.htm",
    "https://gs.shu.edu.cn/private/173242.htm",
])
def test_permanent_detail_redirect_denial_withdraws_only_listing_evidence(
    database, existing_kind, force_refresh, target,
):
    def response(request):
        if force_refresh and "if-none-match" in request.headers:
            return 304, ""
        return 302, "", {"location": target}

    pipeline, client, seen = paginated_pipeline(database, responses={OFFICIAL_URL: response})
    before = None
    if existing_kind == "listing":
        pipeline.notices.upsert_listing(
            DiscoveredNotice("graduate", OFFICIAL_URL, OFFICIAL_TITLE),
            list_url=LIST_ONE, checked_at=datetime(2026, 10, 5, tzinfo=UTC),
        )
    elif existing_kind == "parsed":
        pipeline.notices.upsert(
            NoticeDraft("graduate", OFFICIAL_URL, OFFICIAL_TITLE, OFFICIAL_BODY),
            verification=Verification(ProofStatus.VERIFIED_SNAPSHOT,
                                      datetime(2026, 10, 5, tzinfo=UTC), "snapshot:fixture"),
        )
        before = pipeline.notices.find_by_url(OFFICIAL_URL)
    if force_refresh:
        client.cache.save(OFFICIAL_URL, etag='"fixture"', last_modified=None)

    with client:
        result = pipeline.run().by_source["graduate"]

    assert result.state == "redirect_not_allowed"
    assert target not in seen
    assert seen.count(OFFICIAL_URL) == (2 if force_refresh else 1)
    after = pipeline.notices.find_by_url(OFFICIAL_URL)
    if existing_kind == "parsed":
        assert after == before
        assert len(pipeline.notices.detail_evidence_verified(after.id)["versions"]) == 1
    else:
        assert pipeline.notices.list_browseable() == []
        if existing_kind == "listing":
            assert after.proof_status is ProofStatus.ATTACHMENT_EXCLUDED
        else:
            assert after is None


@pytest.mark.parametrize("existing_kind", ["fresh", "listing", "parsed"])
@pytest.mark.parametrize("target", [
    "https://gs.shu.edu.cn/info/1029/173222.htm",
    "https://gs.shu.edu.cn/info/1029/173242.htm?detail=1",
    "https://example.invalid/info/1029/173242.htm",
])
def test_changed_304_location_withdraws_only_listing_evidence(database, existing_kind, target):
    client = OfficialFixtureClient()
    pipeline = official_pipeline(database, client)
    before = None
    if existing_kind == "listing":
        pipeline.notices.upsert_listing(
            DiscoveredNotice("graduate", OFFICIAL_URL, OFFICIAL_TITLE),
            list_url=pipeline.registry["graduate"].config.list_url,
            checked_at=datetime(2026, 10, 5, tzinfo=UTC),
        )
    elif existing_kind == "parsed":
        pipeline.run()
        before = pipeline.notices.find_by_url(OFFICIAL_URL)
    client.modified = True
    client.not_modified_final_url = target

    result = pipeline.run().by_source["graduate"]

    assert result.state == "redirect_not_allowed"
    assert client.force_refreshes == 0
    after = pipeline.notices.find_by_url(OFFICIAL_URL)
    if existing_kind == "parsed":
        assert after == before
    else:
        assert pipeline.notices.list_browseable() == []
        if existing_kind == "listing":
            assert after.proof_status is ProofStatus.ATTACHMENT_EXCLUDED
        else:
            assert after is None


def test_public_contact_changes_do_not_create_redundant_notice_versions(database):
    client = OfficialFixtureClient(body=OFFICIAL_BODY + "联系人：13800138000。")
    pipeline = official_pipeline(database, client)
    assert pipeline.run().saved_count == 1
    client.body = OFFICIAL_BODY + "联系人：13900139000。"
    assert pipeline.run().saved_count == 0
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM notice_versions").fetchone()[0] == 1


def test_source_status_counts_survive_reopen(database):
    SourceRepository(database).record_status({
        "source_id": "computer", "state": "success", "saved_count": 0,
        "updated_count": 0, "skipped_count": 2,
        "skip_reasons": {"sensitive_content": 2},
    })
    latest = SourceRepository(Database(database.path)).latest_statuses()[0]
    assert latest["saved_count"] == 0
    assert latest["updated_count"] == 0
    assert latest["skipped_count"] == 2
    assert latest["skip_reasons"] == {"sensitive_content": 2}


def test_unknown_skip_reason_is_persisted_only_as_other(database):
    SourceRepository(database).record_status({
        "source_id": "computer", "state": "success", "skipped_count": 1,
        "skip_reasons": {"<script>https://example.invalid/</script>": 1},
    })
    latest = SourceRepository(Database(database.path)).latest_statuses()[0]
    assert latest["skip_reasons"] == {"other": 1}
    assert "example.invalid" not in str(latest)


def test_repeated_source_attempt_counts_only_its_own_skips(database):
    report = official_pipeline(database, OfficialFixtureClient(
        body=OFFICIAL_BODY + "学号：2312345678。"
    )).run(["graduate", "graduate"])
    assert report.by_source["graduate"].skipped_count == 1
    assert report.by_source["graduate"].skip_reasons == {"content_unchanged": 1}


def test_exception_text_is_not_persisted_or_reported(database):
    secret = "<b>https://example.invalid/?phone=13800138000</b>"

    class ExplodingClient:
        def fetch(
            self, source, url, *, force_refresh=False, redirect_target_guard=None, purpose="detail",
        ):
            raise RuntimeError(secret)

    report = official_pipeline(database, ExplodingClient()).run()
    status = SourceRepository(Database(database.path)).latest_statuses()[0]
    assert report.by_source["graduate"].state == "error"
    assert secret not in str(report.events)
    assert secret not in str(status)
    assert status["message"] == "error"


@pytest.mark.parametrize("failure,expected_state", [
    (RobotsUnavailable("private-robots-path"), "robots_unavailable"),
    (httpx.ConnectError("private-proxy-address"), "connection_failed"),
    (httpx.ReadTimeout("private-timeout-detail"), "connection_timeout"),
])
def test_failure_before_first_list_page_does_not_claim_pagination_failure(
    database, failure, expected_state,
):
    class FailingClient:
        def fetch(
            self, source, url, *, force_refresh=False, redirect_target_guard=None, purpose="detail",
        ):
            raise failure

    result = official_pipeline(database, FailingClient()).run().by_source["graduate"]
    persisted = SourceRepository(database).latest_statuses()[0]
    assert result.state == expected_state
    assert result.message == expected_state
    assert result.list_pages_read == 0
    assert result.coverage_code == ""
    assert persisted["coverage_code"] == ""
    assert "private-" not in str(persisted)


def test_real_robots_connect_failure_records_safe_stage_without_list_or_card(database):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        raise httpx.ConnectError("private@example.invalid http://proxy.invalid:8080")

    with PoliteHttpClient(
        contact="", transport=httpx.MockTransport(handler), sleep=lambda _: None,
    ) as client:
        pipeline = official_pipeline(database, client)
        result = pipeline.run().by_source["graduate"]

    status = SourceRepository(database).latest_statuses()[0]
    assert seen == ["https://gs.shu.edu.cn/robots.txt"]
    assert (result.state, result.message) == ("connection_failed", "transport_robots_connect")
    assert result.list_pages_read == 0 and result.request_count == 1
    assert result.request_breakdown == {"robots": 1, "list": 0, "detail": 0, "retry": 0}
    assert (status["state"], status["message"]) == (result.state, result.message)
    assert pipeline.notices.list_browseable() == []
    assert "private@example.invalid" not in str(status)
    assert "proxy.invalid" not in database.path.read_text(errors="ignore")


def test_real_list_read_timeout_retains_retry_and_records_safe_stage(database):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /",
                                  headers={"content-type": "text/plain"})
        raise httpx.ReadTimeout("private@example.invalid http://proxy.invalid:8080")

    with PoliteHttpClient(
        contact="", transport=httpx.MockTransport(handler), sleep=lambda _: None,
    ) as client:
        pipeline = official_pipeline(database, client)
        result = pipeline.run().by_source["graduate"]

    status = SourceRepository(database).latest_statuses()[0]
    assert seen == ["https://gs.shu.edu.cn/robots.txt"] + [
        "https://gs.shu.edu.cn/xwlb/sy.htm",
    ] * 2
    assert (result.state, result.message) == ("connection_timeout", "transport_list_read_timeout")
    assert result.list_pages_read == 0 and result.request_count == 3
    assert result.request_breakdown == {"robots": 1, "list": 1, "detail": 0, "retry": 1}
    assert status["message"] == result.message
    assert pipeline.notices.list_browseable() == []
    assert "private@example.invalid" not in str(status)


def test_real_detail_proxy_failure_keeps_queue_and_other_source_proceeds(database):
    registry = load_source_registry(Path("config/sources.toml"))
    selected = {source_id: registry[source_id] for source_id in (
        "graduate", "undergraduate",
    )}
    fixture = FixtureClient()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /",
                                  headers={"content-type": "text/plain"})
        if str(request.url) == SAFE_DETAIL_URL:
            raise httpx.ProxyError("private@example.invalid http://proxy.invalid:8080")
        source_id = "graduate" if request.url.host == "gs.shu.edu.cn" else "undergraduate"
        source = selected[source_id].config
        if source_id == "graduate" and str(request.url) == source.list_url:
            body = graduate_listing([(SAFE_DETAIL_URL, OFFICIAL_TITLE)],
                                    published="2026/09/11")
            return httpx.Response(200, text=body, headers={"content-type": "text/html"})
        page = fixture.fetch(source, str(request.url))
        return httpx.Response(page.status_code, text=page.body,
                              headers={"content-type": page.content_type})

    with PoliteHttpClient(
        contact="", transport=httpx.MockTransport(handler), sleep=lambda _: None,
    ) as client:
        pipeline = IngestionPipeline(
            registry=selected, client=client,
            notices=NoticeRepository(database, OfficialNoticePolicy(
                {source_id: adapter.config for source_id, adapter in selected.items()},
            )),
            sources=SourceRepository(database), today=lambda: date(2026, 10, 6),
        )
        report = pipeline.run(["graduate", "undergraduate"])

    graduate = report.by_source["graduate"]
    undergraduate = report.by_source["undergraduate"]
    statuses = {row["source_id"]: row for row in SourceRepository(database).latest_statuses()}
    assert graduate.state == "partial" and graduate.message == "transport_detail_proxy"
    assert graduate.list_pages_read == 1 and graduate.pending_count == 1
    assert pipeline.pending.urls("graduate") == {SAFE_DETAIL_URL}
    listing = NoticeRepository(database).find_by_url(SAFE_DETAIL_URL)
    assert listing is not None and listing.proof_status is ProofStatus.VERIFIED_LISTING
    assert undergraduate.state == "success" and undergraduate.saved_count == 1
    assert statuses["graduate"]["message"] == "transport_detail_proxy"
    assert "private@example.invalid" not in str(statuses)
    assert "proxy.invalid" not in database.path.read_text(errors="ignore")


def test_http_5xx_detail_remains_temporary_failure_not_transport_diagnosis(database):
    source = load_source_registry(Path("config/sources.toml"))["graduate"].config

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /",
                                  headers={"content-type": "text/plain"})
        if str(request.url) == source.list_url:
            return httpx.Response(200, text=graduate_listing([(SAFE_DETAIL_URL, OFFICIAL_TITLE)]),
                                  headers={"content-type": "text/html"})
        return httpx.Response(503, text="private@example.invalid",
                              headers={"content-type": "text/html"})

    with PoliteHttpClient(
        contact="", transport=httpx.MockTransport(handler), sleep=lambda _: None,
    ) as client:
        pipeline = official_pipeline(database, client)
        result = pipeline.run().by_source["graduate"]

    assert result.state == "partial" and result.message == "temporary_failure"
    assert result.pending_count == 1 and pipeline.pending.urls("graduate") == {SAFE_DETAIL_URL}
    assert SourceRepository(database).latest_statuses()[0]["message"] == "temporary_failure"


def test_robots_status_reason_is_persisted_without_response_body(database):
    def robots_only(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://gs.shu.edu.cn/robots.txt"
        return httpx.Response(503, text="private-response-marker")

    with PoliteHttpClient(
        contact="", transport=httpx.MockTransport(robots_only), sleep=lambda _: None,
    ) as client:
        result = official_pipeline(database, client).run().by_source["graduate"]

    persisted = SourceRepository(database).latest_statuses()[0]
    assert result.state == "robots_server_error"
    assert result.message == "robots_server_error"
    assert result.list_pages_read == 0
    assert result.request_count == 1
    assert persisted["state"] == "robots_server_error"
    assert persisted["message"] == "robots_server_error"
    assert "private-response-marker" not in str(persisted)


def test_missing_robots_allows_verification_and_records_protocol_note(database):
    source = load_source_registry(Path("config/sources.toml"))["graduate"].config
    fixture = OfficialFixtureClient()

    def public_pages(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404, text="not found")
        page = fixture.fetch(source, str(request.url))
        return httpx.Response(page.status_code, text=page.body,
                              headers={"content-type": page.content_type})

    with PoliteHttpClient(contact="", transport=httpx.MockTransport(public_pages),
                          sleep=lambda _: None) as client:
        result = official_pipeline(database, client).run().by_source["graduate"]

    saved = NoticeRepository(database).find_by_url(OFFICIAL_URL)
    persisted = SourceRepository(database).latest_statuses()[0]
    assert result.state == "success" and result.saved_count == 1
    assert result.message == "robots_not_published"
    assert result.request_count == 3 and result.list_pages_read == 1
    assert saved is not None and saved.proof_status is ProofStatus.VERIFIED_LIVE
    assert persisted["message"] == "robots_not_published"


def test_missing_article_container_has_no_whole_page_fallback(database):
    report = official_pipeline(database, OfficialFixtureClient(container=False)).run()
    assert report.saved_count == 0
    record = NoticeRepository(database).find_by_url(OFFICIAL_URL)
    assert record.proof_status is ProofStatus.VERIFIED_LISTING and record.summary == ""
    assert any(event.message == "structure_changed" for event in report.events)


def test_304_does_not_promote_unverified_record_without_fresh_body(database):
    NoticeRepository(database).upsert(
        NoticeDraft("graduate", OFFICIAL_URL, OFFICIAL_TITLE, "尚未核实的旧正文")
    )
    client = OfficialFixtureClient(modified=True)
    report = official_pipeline(database, client).run()
    assert report.saved_count == 1
    assert client.force_refreshes == 1
    saved = NoticeRepository(database).find_by_url(OFFICIAL_URL)
    assert saved is not None and saved.proof_status is ProofStatus.VERIFIED_LIVE


def test_304_refreshes_existing_live_proof_without_new_version(database):
    client = OfficialFixtureClient()
    pipeline = official_pipeline(database, client)
    assert pipeline.run().saved_count == 1
    before = NoticeRepository(database).find_by_url(OFFICIAL_URL)
    client.modified = True
    client.fetched_at = datetime(2026, 9, 26, tzinfo=UTC)
    report = pipeline.run()
    after = NoticeRepository(database).find_by_url(OFFICIAL_URL)
    assert report.saved_count == 0
    assert after.proof_status is ProofStatus.VERIFIED_LIVE
    assert after.verification_ref == before.verification_ref
    assert after.verified_at == datetime(2026, 9, 26, tzinfo=UTC)
    assert after.content_fingerprint == before.content_fingerprint
    assert client.force_refreshes == 0


def test_304_for_other_article_does_not_refresh_existing_proof(database):
    client = OfficialFixtureClient()
    pipeline = official_pipeline(database, client)
    assert pipeline.run().saved_count == 1
    before = NoticeRepository(database).find_by_url(OFFICIAL_URL)
    client.modified = True
    client.fetched_at = datetime(2026, 9, 26, tzinfo=UTC)
    client.not_modified_final_url = "https://gs.shu.edu.cn/info/1029/173222.htm"
    pipeline.run()
    after = NoticeRepository(database).find_by_url(OFFICIAL_URL)
    assert after.verified_at == before.verified_at


def test_title_only_login_page_stops_source_without_saving(database):
    class LoginFixture(OfficialFixtureClient):
        def fetch(
            self, source, url, *, force_refresh=False, redirect_target_guard=None, purpose="detail",
        ):
            page = super().fetch(source, url, force_refresh=force_refresh)
            if url == source.list_url:
                return page
            return FetchedPage(
                source.id, url, url, 200, "text/html",
                "<html><head><title>用户登录</title></head><body>登录</body></html>",
                self.fetched_at,
            )

    report = official_pipeline(database, LoginFixture()).run()
    assert report.by_source["graduate"].state == "login_or_captcha"
    assert NoticeRepository(database).find_by_url(OFFICIAL_URL) is None


def test_unsolicited_304_cannot_refresh_live_verification(database):
    initial = OfficialFixtureClient()
    assert official_pipeline(database, initial).run().saved_count == 1
    before = NoticeRepository(database).find_by_url(OFFICIAL_URL)
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200, text="User-agent: *\nAllow: /",
                headers={"content-type": "text/plain"},
            )
        if str(request.url) == "https://gs.shu.edu.cn/xwlb/sy.htm":
            return httpx.Response(
                200,
                text=graduate_listing([(OFFICIAL_URL, OFFICIAL_TITLE)]),
                headers={"content-type": "text/html"},
            )
        assert str(request.url) == OFFICIAL_URL
        return httpx.Response(304)

    with PoliteHttpClient(
        contact="course@example.com", transport=httpx.MockTransport(handler),
        monotonic=lambda: 0.0, sleep=lambda _: None,
    ) as client:
        report = official_pipeline(database, client).run()
    after = NoticeRepository(database).find_by_url(OFFICIAL_URL)
    assert report.by_source["graduate"].state == "error"
    assert after.verified_at == before.verified_at
    assert not any("If-None-Match" in request.headers for request in seen)


def paginated_pipeline(database, *, links=None, responses=None, reviewed=True,
                       robots="User-agent: *\nAllow: /"):
    """Synthetic pagination contract; not evidence of a school site's current DOM."""
    registry = load_source_registry(Path("config/sources.toml"))
    graduate = registry["graduate"]
    if reviewed:
        graduate.config = replace(
            graduate.config, pagination_selector="a.next",
            pagination_url_pattern=r"/xwlb/sy(?:\d+)?\.htm",
            pagination_reviewed_on=date(2026, 10, 5),
            pagination_link_text=None,
        )
    else:
        graduate.config = replace(
            graduate.config, pagination_selector=None, pagination_url_pattern=None,
            pagination_reviewed_on=None, pagination_link_text=None,
        )
    links = links or {}
    responses = responses or {}
    seen = []

    def handler(request):
        url = str(request.url)
        seen.append(url)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=robots,
                                  headers={"content-type": "text/plain"})
        if url in responses:
            response = responses[url]
            status, body, *extra_headers = response(request) if callable(response) else response
        elif request.url.path.startswith("/xwlb/sy"):
            status = 200
            body = graduate_listing([(OFFICIAL_URL, OFFICIAL_TITLE)])
            if url in links:
                body += f'<a class="next" href="{links[url]}">下一页</a>'
        else:
            assert request.url.path.startswith("/info/1029/")
            status = 200
            body = f'<h1>{OFFICIAL_TITLE}</h1><div class="v_news_content">{OFFICIAL_BODY}</div>'
        headers = {"content-type": "text/html"}
        if url in responses and extra_headers:
            headers.update(extra_headers[0])
        return httpx.Response(status, text=body, headers=headers)

    client = PoliteHttpClient(contact="offline@example.com", transport=httpx.MockTransport(handler),
                              monotonic=lambda: 0.0, sleep=lambda _: None)
    pipeline = IngestionPipeline(
        registry={"graduate": graduate}, client=client,
        notices=NoticeRepository(database), sources=SourceRepository(database),
        today=lambda: date(2026, 10, 6),
    )
    return pipeline, client, seen


LIST_ONE = "https://gs.shu.edu.cn/xwlb/sy.htm"
LIST_TWO = "https://gs.shu.edu.cn/xwlb/sy2.htm"
LIST_THREE = "https://gs.shu.edu.cn/xwlb/sy3.htm"
LIST_FOUR = "https://gs.shu.edu.cn/xwlb/sy4.htm"


@pytest.mark.parametrize("reviewed,target", [
    (False, "https://gs.shu.edu.cn/info/1029/900001.htm"),
    (False, LIST_TWO),
    (True, "https://gs.shu.edu.cn/info/1029/900001.htm"),
    (True, "https://gs.shu.edu.cn/xwlb/sy/alternative.htm"),
])
@pytest.mark.parametrize("forced_refresh", [False, True])
def test_list_redirect_rejects_unreviewed_target_before_physical_request(
    database, reviewed, target, forced_refresh,
):
    def response(request):
        if "if-none-match" in request.headers:
            return (304, "")
        return (302, "", {"location": target})

    listing = graduate_listing([(OFFICIAL_URL, OFFICIAL_TITLE)])
    pipeline, client, seen = paginated_pipeline(database, reviewed=reviewed, responses={
        LIST_ONE: response, target: (200, listing),
    })
    if forced_refresh:
        client.cache.save(LIST_ONE, etag='"fixture"', last_modified=None)
    with client:
        result = pipeline.run().by_source["graduate"]
    assert target not in seen
    assert seen == ["https://gs.shu.edu.cn/robots.txt", LIST_ONE] + (
        [LIST_ONE] if forced_refresh else [])
    assert result.state != "success"
    assert result.saved_count == result.list_pages_read == 0
    assert result.coverage_code == "pagination_not_allowed"
    assert result.request_count == len(seen)
    assert pipeline.notices.count() == 0
    assert pipeline.sources.latest_completed_checks() == []


def test_reviewed_list_redirect_is_allowed_but_intermediate_cycle_is_not_repeated(database):
    pipeline, client, seen = paginated_pipeline(database, responses={
        LIST_ONE: (302, "", {"location": LIST_TWO}),
        LIST_TWO: (302, "", {"location": LIST_THREE}),
        LIST_THREE: (302, "", {"location": LIST_TWO}),
    })
    with client:
        result = pipeline.run().by_source["graduate"]
    assert seen == ["https://gs.shu.edu.cn/robots.txt", LIST_ONE, LIST_TWO, LIST_THREE]
    assert result.message == "pagination_loop"
    assert result.saved_count == 0


def test_allowed_reviewed_list_redirect_parses_reviewed_destination(database):
    pipeline, client, seen = paginated_pipeline(database, responses={
        LIST_ONE: (302, "", {"location": LIST_TWO}),
    })
    with client:
        result = pipeline.run().by_source["graduate"]
    assert result.state == "success"
    assert result.list_pages_read == result.saved_count == 1
    assert seen == ["https://gs.shu.edu.cn/robots.txt", LIST_ONE, LIST_TWO, OFFICIAL_URL]


def test_unreviewed_returned_list_location_is_checked_even_with_alternate_client(pipeline):
    class ChangedLocation(FixtureClient):
        def fetch(self, source, url, **kwargs):
            return replace(super().fetch(source, url, **kwargs), final_url=LIST_TWO)

    pipeline.client = ChangedLocation()
    result = pipeline.run(["graduate"]).by_source["graduate"]
    assert result.list_pages_read == result.saved_count == 0
    assert result.coverage_code == "pagination_not_allowed"
    assert pipeline.client.requested_urls == [LIST_ONE]


def test_list_traversal_stops_after_five_reviewed_pages_and_dedupes_details(database):
    fifth = "https://gs.shu.edu.cn/xwlb/sy5.htm"
    sixth = "https://gs.shu.edu.cn/xwlb/sy6.htm"
    pipeline, client, seen = paginated_pipeline(database, links={
        LIST_ONE: LIST_TWO, LIST_TWO: LIST_THREE, LIST_THREE: LIST_FOUR,
        LIST_FOUR: fifth, fifth: sixth,
    })
    with client:
        report = pipeline.run()
    result = report.by_source["graduate"]
    assert result.list_pages_read == 5
    assert result.coverage_code == "page_limit_reached"
    assert result.state == "partial"
    assert result.saved_count == 1
    assert seen[:6] == ["https://gs.shu.edu.cn/robots.txt", LIST_ONE, LIST_TWO,
                        LIST_THREE, LIST_FOUR, fifth]
    assert sixth not in seen
    assert seen.count(OFFICIAL_URL) == 1
    assert result.request_count == 7  # robots, five lists, one detail


def test_reviewed_list_range_can_finish_before_page_limit(database):
    pipeline, client, seen = paginated_pipeline(database, links={LIST_ONE: LIST_TWO})
    with client:
        report = pipeline.run()
    result = report.by_source["graduate"]
    assert result.state == "success"
    assert result.list_pages_read == 2
    assert result.coverage_code == "reviewed_range_complete"
    assert seen.count(OFFICIAL_URL) == 1


@pytest.mark.parametrize("next_url", [LIST_ONE, LIST_TWO])
def test_list_cycles_are_not_requested_again(database, next_url):
    pipeline, client, seen = paginated_pipeline(database, links={
        LIST_ONE: LIST_TWO, LIST_TWO: next_url,
    })
    with client:
        report = pipeline.run()
    result = report.by_source["graduate"]
    assert result.state == "partial"
    assert result.coverage_code in {"pagination_loop", "pagination_failed"}
    assert seen.count(LIST_ONE) == 1
    assert seen.count(LIST_TWO) == 1
    assert OFFICIAL_URL not in seen  # stop immediately once traversal is unsafe


def test_third_page_cycle_cannot_be_mistaken_for_clean_page_limit(database):
    pipeline, client, seen = paginated_pipeline(database, links={
        LIST_ONE: LIST_TWO, LIST_TWO: LIST_THREE, LIST_THREE: LIST_ONE,
    })
    with client:
        result = pipeline.run().by_source["graduate"]
    assert result.state == "partial"
    assert result.coverage_code == "pagination_loop"
    assert result.list_pages_read == 3
    assert OFFICIAL_URL not in seen


@pytest.mark.parametrize("target", [LIST_ONE, LIST_ONE + "?utm_source=loop#top"])
def test_list_redirect_to_visited_page_stops_before_repeat_http_attempt(database, target):
    pipeline, client, seen = paginated_pipeline(database, links={LIST_ONE: LIST_TWO},
                                              responses={LIST_TWO: (302, "", {"location": target})})
    with client:
        result = pipeline.run().by_source["graduate"]
    assert seen == ["https://gs.shu.edu.cn/robots.txt", LIST_ONE, LIST_TWO]
    assert result.state == "partial"
    assert result.message == "pagination_loop"
    assert result.coverage_code == "pagination_loop"
    assert result.list_pages_read == 1
    assert result.request_count == 3
    assert result.saved_count == 0


@pytest.mark.parametrize("next_url", ["https://outside.example/sy2.htm", "/info/1/2.htm"])
def test_unsafe_pagination_never_requests_selected_url(database, next_url):
    pipeline, client, seen = paginated_pipeline(database, links={LIST_ONE: next_url})
    with client:
        report = pipeline.run()
    result = report.by_source["graduate"]
    assert result.state == "partial"
    assert result.coverage_code == "pagination_failed"
    assert len(seen) == 2  # robots and first list only
    assert result.list_pages_read == 1


@pytest.mark.parametrize("status,body,message", [
    (403, "", "forbidden"), (429, "", "rate_limited"),
    (200, "<title>用户登录</title>", "login_or_captcha"),
    (200, "<main>changed list markup</main>", "structure_changed"),
    (500, "", "http_error"),
])
def test_second_page_failure_preserves_list_progress_without_finishing(
    database, status, body, message,
):
    pipeline, client, seen = paginated_pipeline(
        database, links={LIST_ONE: LIST_TWO}, responses={LIST_TWO: (status, body)},
    )
    with client:
        report = pipeline.run()
    result = report.by_source["graduate"]
    assert result.state == "partial"
    assert result.message == message
    assert result.list_pages_read == 1
    assert result.coverage_code == "list_check_incomplete"
    assert OFFICIAL_URL not in seen
    assert not any(event.kind == "source_finished" for event in report.events)
    listing = pipeline.notices.list_browseable()[0]
    assert listing.canonical_url == OFFICIAL_URL and listing.summary == ""
    assert listing.proof_status is ProofStatus.VERIFIED_LISTING
    assert pipeline.notices.list_verified() == []


def test_unreviewed_pagination_reads_only_first_page_and_exposes_coverage(database):
    pipeline, client, seen = paginated_pipeline(
        database, reviewed=False, links={LIST_ONE: LIST_TWO},
    )
    with client:
        result = pipeline.run().by_source["graduate"]
    assert result.list_pages_read == 1
    assert result.coverage_code == "pagination_unverified"
    assert LIST_TWO not in seen


def test_list_stats_round_trip_with_caller_owned_run_id(database):
    pipeline, client, _ = paginated_pipeline(database, links={LIST_ONE: LIST_TWO})
    run_id = ManualUpdateRunRepository(database).start()
    with client:
        pipeline.run(run_id=run_id)
    status = SourceRepository(database).latest_statuses()[0]
    assert status["run_id"] == run_id
    assert status["list_pages_read"] == 2
    assert status["request_count"] == 4
    assert status["coverage_code"] == "reviewed_range_complete"


def test_second_list_page_rechecks_robots_before_http_attempt(database):
    pipeline, client, seen = paginated_pipeline(
        database, links={LIST_ONE: LIST_TWO},
        robots="User-agent: *\nDisallow: /xwlb/sy2.htm",
    )
    with client:
        result = pipeline.run().by_source["graduate"]
    assert result.state == "partial"
    assert result.message == "robots_disallowed"
    assert result.list_pages_read == 1
    assert result.request_count == 2
    assert LIST_TWO not in seen
    assert OFFICIAL_URL not in seen
    assert SourceRepository(database).latest_statuses()[0]["message"] == "robots_disallowed"


def test_candidate_quota_keeps_saved_notices_without_claiming_physical_exhaustion(database):
    body = graduate_listing([
        (f'/info/1029/{number}.htm', OFFICIAL_TITLE)
        for number in range(173242, 173276)
    ])
    pipeline, client, seen = paginated_pipeline(database, responses={LIST_ONE: (200, body)})
    with client:
        result = pipeline.run().by_source["graduate"]
    assert result.state == "partial"
    assert result.message == "candidate_limit_reached"
    assert result.coverage_code == "reviewed_range_complete"
    assert result.request_count == 34
    assert result.list_pages_read == 1
    assert result.saved_count == 32
    assert result.queued_count == 0 and result.pending_count == 2
    assert result.listing_count == 2
    assert NoticeRepository(database).count() == 34
    assert len(pipeline.notices.list_verified()) == 32
    assert len(pipeline.notices.list_browseable()) == 34
    assert len(seen) == 34
    assert "https://gs.shu.edu.cn/info/1029/173275.htm" not in seen


def test_detail_structure_failure_skips_only_bad_article_and_keeps_following_notice(database):
    second_url = "https://gs.shu.edu.cn/info/1029/173243.htm"
    third_url = "https://gs.shu.edu.cn/info/1029/173244.htm"
    body = graduate_listing([
        (url, OFFICIAL_TITLE) for url in (OFFICIAL_URL, second_url, third_url)
    ])
    pipeline, client, seen = paginated_pipeline(database, responses={
        LIST_ONE: (200, body), second_url: (200, "<h1>通知</h1><main>changed container</main>"),
    })
    with client:
        result = pipeline.run().by_source["graduate"]
    assert result.state == "partial"
    assert result.message == "structure_changed"
    assert result.saved_count == 2
    assert result.skip_reasons == {"structure_changed": 1}
    assert result.list_pages_read == 1
    assert result.request_count == 5
    assert third_url in seen
    assert NoticeRepository(database).count() == 3
    assert len(pipeline.notices.list_verified()) == 2
    assert pipeline.notices.find_by_url(second_url).proof_status is ProofStatus.VERIFIED_LISTING


def test_all_details_with_changed_structure_do_not_claim_success(database):
    second_url = "https://gs.shu.edu.cn/info/1029/173243.htm"
    body = graduate_listing([
        (url, OFFICIAL_TITLE) for url in (OFFICIAL_URL, second_url)
    ])
    changed = (200, "<h1>通知</h1><main>changed container</main>")
    pipeline, client, seen = paginated_pipeline(database, responses={
        LIST_ONE: (200, body), OFFICIAL_URL: changed, second_url: changed,
    })
    with client:
        result = pipeline.run().by_source["graduate"]
    assert result.state == "partial"
    assert result.message == "structure_changed"
    assert result.saved_count == 0
    assert result.skip_reasons == {"structure_changed": 2}
    assert seen == ["https://gs.shu.edu.cn/robots.txt", LIST_ONE, OFFICIAL_URL, second_url]
    assert len(pipeline.notices.list_browseable()) == 2
    assert pipeline.notices.list_verified() == []


def test_iframe_only_detail_is_explicitly_skipped_as_attachment(database):
    body = graduate_listing([(OFFICIAL_URL, OFFICIAL_TITLE)])
    detail = (
        f"<form><table><tr><td class='NewsTitle'>{OFFICIAL_TITLE}</td></tr></table>"
        "<div class='v_news_content'>"
        "<iframe src='/system/pdfjs/viewer.html'></iframe>"
        "</div></form>"
    )
    pipeline, client, _seen = paginated_pipeline(database, responses={
        LIST_ONE: (200, body), OFFICIAL_URL: (200, detail),
    })

    with client:
        result = pipeline.run().by_source["graduate"]

    assert result.state == "success"
    assert result.saved_count == 0
    assert result.skip_reasons == {"attachment_blocked": 1}
    assert NoticeRepository(database).count() == 0


@pytest.mark.parametrize("failure", [403, 429, "robots", "structure", "login"])
def test_source_hard_stops_leave_other_source_independent(database, failure):
    from shu_insight.ingest.http_client import RobotsDisallowed

    class FailedSourceClient(FixtureClient):
        def fetch(
            self, source, url, *, force_refresh=False, redirect_target_guard=None, purpose="detail",
        ):
            if source.id == "undergraduate":
                if failure == "robots":
                    raise RobotsDisallowed("robots_disallowed")
                page = super().fetch(source, url, force_refresh=force_refresh)
                return replace(page, status_code=failure if isinstance(failure, int) else 200,
                               body="<title>用户登录</title>" if failure == "login" else "")
            return super().fetch(source, url, force_refresh=force_refresh)

    registry = load_source_registry(Path("config/sources.toml"))
    pipeline = IngestionPipeline(
        registry={key: registry[key] for key in ("undergraduate", "graduate")},
        client=FailedSourceClient(), notices=NoticeRepository(database),
        sources=SourceRepository(database),
        today=lambda: date(2026, 10, 6),
    )
    report = pipeline.run()
    assert report.by_source["undergraduate"].state != "success"
    assert report.by_source["graduate"].state == "success"
    assert report.by_source["graduate"].saved_count == 1


@pytest.mark.parametrize("age,state", [(89, "success"), (90, "config_review_required")])
def test_source_review_expiry_boundary_skips_only_stale_source(database, age, state):
    registry = load_source_registry(Path("config/sources.toml"))
    registry["undergraduate"].config = replace(
        registry["undergraduate"].config, reviewed_on=date(2026, 7, 8),
    )
    client = FixtureClient()
    pipeline = IngestionPipeline(
        registry={key: registry[key] for key in ("undergraduate", "graduate")},
        client=client, notices=NoticeRepository(database), sources=SourceRepository(database),
        today=lambda: date(2026, 10, 5 if age == 89 else 6),
    )
    report = pipeline.run()
    assert report.by_source["undergraduate"].state == state
    assert report.by_source["graduate"].saved_count == 1
    if age == 90:
        assert not any("bksy.shu.edu.cn" in url for url in client.requested_urls)
        assert report.by_source["undergraduate"].coverage_code == "config_review_required"
