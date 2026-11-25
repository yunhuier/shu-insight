from __future__ import annotations

from datetime import date

import httpx
import pytest

from shu_insight.domain.models import SourceConfig
from shu_insight.ingest.http_client import (
    HttpCache,
    PoliteHttpClient,
    RedirectNotAllowed,
    RequestBudgetExceeded,
    RobotsDisallowed,
    RobotsUnavailable,
    SourceCircuitBreaker,
    SourceCircuitOpen,
)


def source(source_id: str, host: str, path: str) -> SourceConfig:
    return SourceConfig(
        id=source_id,
        name=source_id,
        base_url=f"https://{host}/",
        list_url=f"https://{host}{path}",
        allowed_path_prefixes=(path.rsplit("/", 1)[0], "/info/"),
        authority_score=0.9,
        minimum_interval_seconds=2.0,
        reviewed_on=date(2026, 9, 21),
    )


SHU_SOURCE = source("shu_main", "www.shu.edu.cn", "/tzgg.htm")
BKSY_SOURCE = source("undergraduate", "bksy.shu.edu.cn", "/index/tzgg.htm")


class RecordingTransport:
    def __init__(self) -> None:
        self.responses: dict[str, list[httpx.Response | Exception]] = {}
        self.requests: list[httpx.Request] = []

    def add(
        self,
        url: str,
        *,
        status: int = 200,
        body: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.responses.setdefault(url, []).append(
            httpx.Response(status, text=body, headers=headers)
        )

    def add_exception(self, url: str, error: Exception) -> None:
        self.responses.setdefault(url, []).append(error)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        queued = self.responses.get(str(request.url), [])
        if not queued:
            raise AssertionError(f"unexpected request: {request.url}")
        response = queued.pop(0)
        if isinstance(response, Exception):
            raise response
        response.request = request
        return response

    def request_count(self, url: str) -> int:
        return sum(str(request.url) == url for request in self.requests)


class Clock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


def make_client(
    transport: RecordingTransport,
    *,
    cache: HttpCache | None = None,
    breaker: SourceCircuitBreaker | None = None,
    clock: Clock | None = None,
) -> PoliteHttpClient:
    clock = clock or Clock()
    return PoliteHttpClient(
        transport=httpx.MockTransport(transport),
        cache=cache or HttpCache(),
        breaker=breaker or SourceCircuitBreaker(),
        contact="student@example.com",
        monotonic=clock.now,
        sleep=clock.sleep,
    )


def allow_robots(transport: RecordingTransport, source_config: SourceConfig) -> None:
    transport.add(
        f"{source_config.base_url}robots.txt",
        body="User-agent: *\nAllow: /",
    )


def test_blank_contact_uses_identifiable_user_agent_without_personal_email() -> None:
    transport = RecordingTransport()
    allow_robots(transport, SHU_SOURCE)
    transport.add(SHU_SOURCE.list_url, body="notice list")

    with PoliteHttpClient(
        contact="", transport=httpx.MockTransport(transport), sleep=lambda _: None,
    ) as client:
        page = client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)

    assert page.status_code == 200
    assert transport.request_count(SHU_SOURCE.list_url) == 1
    assert all("SHU-Insight-CourseProject" in request.headers["User-Agent"]
               and "contact=" not in request.headers["User-Agent"]
               for request in transport.requests)


@pytest.mark.parametrize("unsafe_contact", ["中文联系邮箱", "name@example.com\r\nX-Injected: yes"])
def test_unsafe_optional_contact_is_omitted_without_blocking_update(
    unsafe_contact: str,
) -> None:
    transport = RecordingTransport()
    allow_robots(transport, SHU_SOURCE)
    transport.add(SHU_SOURCE.list_url, body="notice list")

    with PoliteHttpClient(
        contact=unsafe_contact, transport=httpx.MockTransport(transport), sleep=lambda _: None,
    ) as client:
        page = client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)

    assert page.status_code == 200
    assert all("contact=" not in request.headers["User-Agent"]
               and "X-Injected" not in request.headers for request in transport.requests)


def test_429_opens_only_that_source_circuit() -> None:
    transport = RecordingTransport()
    breaker = SourceCircuitBreaker()
    allow_robots(transport, BKSY_SOURCE)
    transport.add(BKSY_SOURCE.list_url, status=429)
    client = make_client(transport, breaker=breaker)

    page = client.fetch(BKSY_SOURCE, BKSY_SOURCE.list_url)

    assert page.status_code == 429
    assert breaker.allow("undergraduate") is False
    assert breaker.allow("computer") is True


def test_cached_etag_is_sent_and_304_has_no_cached_body() -> None:
    transport = RecordingTransport()
    cache = HttpCache()
    cache.save(SHU_SOURCE.list_url, etag='"abc"', last_modified=None, body="old")
    allow_robots(transport, SHU_SOURCE)
    transport.add(SHU_SOURCE.list_url, status=304)
    client = make_client(transport, cache=cache)

    page = client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)

    page_request = next(
        request for request in transport.requests if str(request.url) == SHU_SOURCE.list_url
    )
    assert page_request.headers["If-None-Match"] == '"abc"'
    assert page.body == ""
    assert page.from_cache is True


def test_robots_disallow_prevents_page_request() -> None:
    transport = RecordingTransport()
    transport.add(
        "https://www.shu.edu.cn/robots.txt",
        body="User-agent: *\nDisallow: /tzgg.htm",
    )
    client = make_client(transport)

    with pytest.raises(RobotsDisallowed):
        client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)

    assert transport.request_count(SHU_SOURCE.list_url) == 0


def test_unavailable_robots_skips_live_access_conservatively() -> None:
    transport = RecordingTransport()
    transport.add("https://www.shu.edu.cn/robots.txt", status=503)
    client = make_client(transport)

    with pytest.raises(RobotsUnavailable):
        client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)

    assert transport.request_count(SHU_SOURCE.list_url) == 0


@pytest.mark.parametrize(("status", "body", "content_type", "expected"), [
    (403, "private-page-marker", "text/html", "robots_forbidden"),
    (429, "private-page-marker", "text/html", "robots_rate_limited"),
    (503, "private-page-marker", "text/plain", "robots_server_error"),
    (302, "private-page-marker", "text/plain", "robots_redirect"),
    (200, "User-agent: *\nAllow: /", "text/html", "robots_wrong_content_type"),
    (200, "<html>private-page-marker</html>", "text/plain", "robots_html_or_login"),
    (200, "User-agent: *\nCrawl-delay: 2", "text/plain", "robots_missing_directives"),
])
def test_unusable_robots_reports_safe_specific_reason_without_list_fetch(
    status: int, body: str, content_type: str, expected: str,
) -> None:
    transport = RecordingTransport()
    transport.add(
        "https://www.shu.edu.cn/robots.txt", status=status, body=body,
        headers={"content-type": content_type},
    )

    with make_client(transport) as client, pytest.raises(RobotsUnavailable) as failure:
        client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)

    assert failure.value.reason_code == expected
    assert body not in str(failure.value)
    assert transport.request_count(SHU_SOURCE.list_url) == 0
    assert transport.request_count("https://www.shu.edu.cn/robots.txt") == 1


@pytest.mark.parametrize("status", [404, 410])
def test_missing_robots_continues_public_requests_with_pacing_and_cached_check(status):
    transport = RecordingTransport()
    transport.add("https://www.shu.edu.cn/robots.txt", status=status, body="not found")
    transport.add(SHU_SOURCE.list_url, body="notice list")
    transport.add(SHU_SOURCE.list_url, body="notice list")
    clock = Clock()
    with make_client(transport, clock=clock) as client:
        assert client.fetch(SHU_SOURCE, SHU_SOURCE.list_url).status_code == 200
        assert client.fetch(SHU_SOURCE, SHU_SOURCE.list_url).status_code == 200
        assert client.robots_note(SHU_SOURCE.id) == "robots_not_published"
        assert client.request_count(SHU_SOURCE.id) == 3
    assert transport.request_count("https://www.shu.edu.cn/robots.txt") == 1
    assert clock.sleeps == [2.0, 2.0]


@pytest.mark.parametrize("status", [403, 429])
def test_missing_robots_does_not_override_public_page_access_stop(status):
    transport = RecordingTransport()
    transport.add("https://www.shu.edu.cn/robots.txt", status=404)
    transport.add(SHU_SOURCE.list_url, status=status)
    with make_client(transport) as client:
        assert client.fetch(SHU_SOURCE, SHU_SOURCE.list_url).status_code == status
        with pytest.raises(SourceCircuitOpen):
            client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)
    assert transport.request_count(SHU_SOURCE.list_url) == 1


def test_requests_on_same_host_observe_minimum_interval() -> None:
    transport = RecordingTransport()
    clock = Clock()
    allow_robots(transport, SHU_SOURCE)
    transport.add(SHU_SOURCE.list_url, body="notice list")
    client = make_client(transport, clock=clock)

    client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)

    assert clock.sleeps == [2.0]


def test_timeout_retries_once_then_succeeds() -> None:
    transport = RecordingTransport()
    allow_robots(transport, SHU_SOURCE)
    transport.add_exception(SHU_SOURCE.list_url, httpx.ReadTimeout("slow"))
    transport.add(SHU_SOURCE.list_url, body="ok")
    client = make_client(transport)

    page = client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)

    assert page.body == "ok"
    assert transport.request_count(SHU_SOURCE.list_url) == 2


def test_two_clients_reuse_persistent_validators_without_persisting_raw_body(tmp_path):
    path = tmp_path / "validators.sqlite3"
    url = "https://www.shu.edu.cn/info/1/2.htm"
    first = RecordingTransport()
    allow_robots(first, SHU_SOURCE)
    first.add(url, body="RAW_DETAIL_NEVER_PERSIST", headers={"etag": '"v1"'})
    with make_client(first, cache=HttpCache(path)) as client:
        assert client.fetch(SHU_SOURCE, url).body == "RAW_DETAIL_NEVER_PERSIST"
    assert b"RAW_DETAIL_NEVER_PERSIST" not in path.read_bytes()
    second = RecordingTransport()
    allow_robots(second, SHU_SOURCE)
    second.add(url, status=304)
    with make_client(second, cache=HttpCache(path)) as client:
        page = client.fetch(SHU_SOURCE, url)
    requests = [request for request in second.requests if str(request.url) == url]
    assert requests[0].headers["If-None-Match"] == '"v1"'
    assert len(requests) == 1
    assert page.body == ""
    assert page.status_code == 304 and page.from_cache


def test_redirect_to_foreign_host_is_never_requested() -> None:
    source_config = source("computer", "cs.shu.edu.cn", "/xxgk/tzgg.htm")
    transport = RecordingTransport()
    allow_robots(transport, source_config)
    transport.add(source_config.list_url, status=302,
                  headers={"location": "https://evil.example/private"})
    with make_client(transport) as client, pytest.raises(RedirectNotAllowed):
        client.fetch(source_config, source_config.list_url)
    assert all(request.url.host != "evil.example" for request in transport.requests)


@pytest.mark.parametrize("bad_url", [
    "https://cs.shu.edu.cn:444/info/1381/50001.htm",
    "https://user@cs.shu.edu.cn/info/1381/50001.htm",
    "https://cs.shu.edu.cn/info/%2e%2e/private.htm",
    "https://cs.shu.edu.cn/info/%252e%252e/private.htm",
])
def test_bad_direct_url_is_rejected_before_network(bad_url: str) -> None:
    source_config = source("computer", "cs.shu.edu.cn", "/xxgk/tzgg.htm")
    transport = RecordingTransport()
    with make_client(transport) as client, pytest.raises(RedirectNotAllowed):
        client.fetch(source_config, bad_url)
    assert transport.requests == []


def test_robots_redirect_stops_before_following() -> None:
    transport = RecordingTransport()
    transport.add("https://www.shu.edu.cn/robots.txt", status=302,
                  headers={"location": "https://evil.example/robots.txt"})
    with make_client(transport) as client, pytest.raises(RobotsUnavailable):
        client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)
    assert len(transport.requests) == 1


def test_robots_specific_user_agent_rule_blocks_request() -> None:
    transport = RecordingTransport()
    transport.add("https://www.shu.edu.cn/robots.txt", body=(
        "User-agent: SHU-Insight-CourseProject\nDisallow: /tzgg.htm\n"
        "User-agent: *\nAllow: /\n"
    ))
    with make_client(transport) as client, pytest.raises(RobotsDisallowed):
        client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)
    assert transport.request_count(SHU_SOURCE.list_url) == 0


def test_same_host_redirect_checks_target_robots_rule_before_get() -> None:
    transport = RecordingTransport()
    transport.add("https://www.shu.edu.cn/robots.txt", body=(
        "User-agent: *\nDisallow: /info/private/\n"
    ))
    transport.add(SHU_SOURCE.list_url, status=302,
                  headers={"location": "/info/private/2.htm"})
    with make_client(transport) as client, pytest.raises(RobotsDisallowed):
        client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)
    assert transport.request_count("https://www.shu.edu.cn/info/private/2.htm") == 0


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        ("<html><title>用户登录</title><body>请登录</body></html>", "text/html"),
        ("Error: service temporarily unavailable", "text/plain"),
        ("<html><title>用户登录</title></html>", "text/plain"),
    ],
)
def test_200_non_robots_response_stops_before_notice_get(body: str, content_type: str) -> None:
    transport = RecordingTransport()
    transport.add(
        "https://www.shu.edu.cn/robots.txt", body=body,
        headers={"content-type": content_type},
    )
    with make_client(transport) as client, pytest.raises(RobotsUnavailable):
        client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)
    assert transport.request_count(SHU_SOURCE.list_url) == 0


def test_robots_empty_disallow_is_usable_and_allows_notice_get() -> None:
    transport = RecordingTransport()
    transport.add(
        "https://www.shu.edu.cn/robots.txt",
        body="User-agent: *\nDisallow:\n",
        headers={"content-type": "text/plain; charset=utf-8"},
    )
    transport.add(SHU_SOURCE.list_url, body="notice list")
    with make_client(transport) as client:
        assert client.fetch(SHU_SOURCE, SHU_SOURCE.list_url).status_code == 200
    assert transport.request_count(SHU_SOURCE.list_url) == 1


def test_unsolicited_304_without_validator_cannot_be_used_as_proof() -> None:
    transport = RecordingTransport()
    allow_robots(transport, SHU_SOURCE)
    transport.add(SHU_SOURCE.list_url, status=304)
    with make_client(transport) as client, pytest.raises(
        RuntimeError, match="unsolicited_not_modified"
    ):
        client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)
    assert transport.request_count(SHU_SOURCE.list_url) == 1


def test_robots_and_notice_requests_count_toward_source_budget() -> None:
    transport = RecordingTransport()
    allow_robots(transport, SHU_SOURCE)
    transport.add(SHU_SOURCE.list_url, body="notice list")
    with make_client(transport) as client:
        assert client.request_count(SHU_SOURCE.id) == 0
        client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)
        assert client.request_count(SHU_SOURCE.id) == 2
        assert client.request_count(BKSY_SOURCE.id) == 0


def test_304_then_forced_refresh_counts_both_physical_requests() -> None:
    transport = RecordingTransport()
    cache = HttpCache()
    cache.save(SHU_SOURCE.list_url, etag='"abc"', last_modified=None)
    allow_robots(transport, SHU_SOURCE)
    transport.add(SHU_SOURCE.list_url, status=304)
    transport.add(SHU_SOURCE.list_url, body="fresh")
    with make_client(transport, cache=cache) as client:
        assert client.fetch(SHU_SOURCE, SHU_SOURCE.list_url).from_cache
        assert client.fetch(SHU_SOURCE, SHU_SOURCE.list_url, force_refresh=True).body == "fresh"
        assert client.request_count(SHU_SOURCE.id) == 3
    assert "If-None-Match" not in transport.requests[-1].headers


def test_allowed_redirect_counts_each_physical_request() -> None:
    transport = RecordingTransport()
    target = "https://www.shu.edu.cn/info/1/2.htm"
    allow_robots(transport, SHU_SOURCE)
    transport.add(SHU_SOURCE.list_url, status=302, headers={"location": target})
    transport.add(target, body="detail")
    with make_client(transport) as client:
        assert client.fetch(SHU_SOURCE, SHU_SOURCE.list_url).final_url == target
        assert client.request_count(SHU_SOURCE.id) == 3


def test_redirect_target_guard_stops_before_target_physical_request() -> None:
    transport = RecordingTransport()
    target = "https://www.shu.edu.cn/info/1/2.htm"
    allow_robots(transport, SHU_SOURCE)
    transport.add(SHU_SOURCE.list_url, status=302, headers={"location": target})
    transport.add(target, body="must never be requested")

    def reject_visited(url):
        if url == target:
            raise RedirectNotAllowed("pagination_loop")

    with make_client(transport) as client, pytest.raises(RedirectNotAllowed,
                                                        match="^pagination_loop$"):
        client.fetch(SHU_SOURCE, SHU_SOURCE.list_url, redirect_target_guard=reject_visited)
    assert transport.request_count(target) == 0
    assert transport.request_count(SHU_SOURCE.list_url) == 1


def test_redirect_target_guard_does_not_skip_robots_checks() -> None:
    transport = RecordingTransport()
    target = "https://www.shu.edu.cn/info/private/2.htm"
    transport.add("https://www.shu.edu.cn/robots.txt",
                  body="User-agent: *\nDisallow: /info/private/")
    transport.add(SHU_SOURCE.list_url, status=302, headers={"location": target})
    transport.add(target, body="must never be requested")
    with make_client(transport) as client, pytest.raises(RobotsDisallowed):
        client.fetch(SHU_SOURCE, SHU_SOURCE.list_url, redirect_target_guard=lambda _: None)
    assert transport.request_count(target) == 0


@pytest.mark.parametrize("error", [httpx.ConnectError("failed"), httpx.ReadTimeout("slow")])
def test_failed_attempt_counts_and_retry_waits_two_seconds(error: Exception) -> None:
    transport = RecordingTransport()
    clock = Clock()
    allow_robots(transport, SHU_SOURCE)
    transport.add_exception(SHU_SOURCE.list_url, error)
    transport.add(SHU_SOURCE.list_url, body="ok")
    with make_client(transport, clock=clock) as client:
        assert client.fetch(SHU_SOURCE, SHU_SOURCE.list_url).body == "ok"
        assert clock.sleeps == [2.0, 2.0]
        assert client.request_count(SHU_SOURCE.id) == 3


@pytest.mark.parametrize("last_attempt_fails", [False, True])
def test_fortieth_attempt_exhausts_only_its_source_budget(last_attempt_fails: bool) -> None:
    transport = RecordingTransport()
    allow_robots(transport, SHU_SOURCE)
    for _ in range(38):
        transport.add(SHU_SOURCE.list_url, body="ok")
    if last_attempt_fails:
        transport.add_exception(SHU_SOURCE.list_url, httpx.ConnectError("private detail"))
    else:
        transport.add(SHU_SOURCE.list_url, body="fortieth")
    # An incorrect extra attempt has a response, so failure proves the missing budget gate.
    transport.add(SHU_SOURCE.list_url, body="must never be requested")
    same_host_source = source("other_source", "www.shu.edu.cn", "/tzgg.htm")
    with make_client(transport) as client:
        for _ in range(38):
            client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)
        if last_attempt_fails:
            with pytest.raises(RequestBudgetExceeded, match="^request_budget_exhausted$") as raised:
                client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)
        else:
            assert client.fetch(SHU_SOURCE, SHU_SOURCE.list_url).body == "fortieth"
            with pytest.raises(RequestBudgetExceeded, match="^request_budget_exhausted$") as raised:
                client.fetch(SHU_SOURCE, SHU_SOURCE.list_url)
        assert raised.value.reason_code == "request_budget_exhausted"
        assert client.request_count(SHU_SOURCE.id) == 40
        assert len(transport.requests) == 40
        assert client.request_breakdown(SHU_SOURCE.id) == {
            "robots": 1, "list": 0, "detail": 39, "retry": 0,
        }
        assert client.fetch(same_host_source, same_host_source.list_url).status_code == 200
        assert client.request_count(same_host_source.id) == 1
        assert client.request_count(SHU_SOURCE.id) == 40


def test_breakdown_counts_retry_redirect_and_304_once_at_physical_boundary():
    transport = RecordingTransport()
    clock = Clock()
    allow_robots(transport, SHU_SOURCE)
    transport.add_exception(SHU_SOURCE.list_url, httpx.ConnectError("private exception"))
    transport.add(SHU_SOURCE.list_url, body="list")
    detail = "https://www.shu.edu.cn/info/1/100.htm"
    target = "https://www.shu.edu.cn/info/1/101.htm"
    transport.add(detail, status=302, headers={"location": target})
    transport.add(target, body="detail", headers={"etag": '"fixture"'})
    transport.add(detail, status=304)
    cache = HttpCache()
    with make_client(transport, clock=clock, cache=cache) as client:
        client.fetch(SHU_SOURCE, SHU_SOURCE.list_url, purpose="list")
        client.fetch(SHU_SOURCE, detail, purpose="detail")
        client.fetch(SHU_SOURCE, detail, purpose="detail")
        assert client.request_breakdown(SHU_SOURCE.id) == {
            "robots": 1, "list": 1, "detail": 3, "retry": 1,
        }
        assert sum(client.request_breakdown(SHU_SOURCE.id).values()) == 6
        assert client.request_count(SHU_SOURCE.id) == len(transport.requests) == 6
        assert clock.sleeps == [2.0] * 5


def test_breakdown_rejects_unknown_purpose_before_any_request():
    transport = RecordingTransport()
    with make_client(transport) as client:
        with pytest.raises(ValueError):
            client.fetch(SHU_SOURCE, SHU_SOURCE.list_url, purpose="private text")
        assert client.request_count(SHU_SOURCE.id) == 0
        assert client.request_breakdown(SHU_SOURCE.id) == {
            "robots": 0, "list": 0, "detail": 0, "retry": 0,
        }


def test_robots_connect_failure_has_safe_stage_without_retry_or_list_request():
    from shu_insight.ingest.http_client import SafeTransportError

    transport = RecordingTransport()
    secret = "private@example.invalid via http://proxy.invalid:8080"
    transport.add_exception(f"{SHU_SOURCE.base_url}robots.txt", httpx.ConnectError(secret))
    transport.add(SHU_SOURCE.list_url, body="must never be requested")
    with make_client(transport) as client, pytest.raises(SafeTransportError) as raised:
        client.fetch(SHU_SOURCE, SHU_SOURCE.list_url, purpose="list")
    assert (raised.value.stage, raised.value.kind) == ("robots", "connect")
    assert secret not in str(raised.value) and secret not in repr(raised.value)
    assert transport.request_count(SHU_SOURCE.list_url) == 0
    assert client.request_count(SHU_SOURCE.id) == 1
    assert client.request_breakdown(SHU_SOURCE.id) == {
        "robots": 1, "list": 0, "detail": 0, "retry": 0,
    }


def test_list_read_timeout_keeps_single_retry_and_safe_stage():
    from shu_insight.ingest.http_client import SafeTransportError

    transport = RecordingTransport()
    clock = Clock()
    allow_robots(transport, SHU_SOURCE)
    for _ in range(2):
        transport.add_exception(SHU_SOURCE.list_url, httpx.ReadTimeout("private@example.invalid"))
    with make_client(transport, clock=clock) as client, pytest.raises(SafeTransportError) as raised:
        client.fetch(SHU_SOURCE, SHU_SOURCE.list_url, purpose="list")
    assert (raised.value.stage, raised.value.kind) == ("list", "read_timeout")
    assert "private@example.invalid" not in str(raised.value)
    assert client.request_breakdown(SHU_SOURCE.id) == {
        "robots": 1, "list": 1, "detail": 0, "retry": 1,
    }
    assert client.request_count(SHU_SOURCE.id) == 3
    assert clock.sleeps == [2.0, 2.0]


@pytest.mark.parametrize(("error", "kind", "retry_count"), [
    (httpx.ProxyError("http://proxy.invalid:8080"), "proxy", 0),
    (httpx.ConnectTimeout("private@example.invalid"), "connect_timeout", 1),
    (httpx.ReadError("private@example.invalid"), "read", 0),
    (httpx.TransportError("private@example.invalid"), "other", 0),
])
def test_detail_transport_failure_exposes_only_safe_kind(error, kind, retry_count):
    from shu_insight.ingest.http_client import SafeTransportError

    transport = RecordingTransport()
    allow_robots(transport, SHU_SOURCE)
    detail_url = "https://www.shu.edu.cn/info/1/100.htm"
    transport.add_exception(detail_url, error)
    if retry_count:
        transport.add_exception(detail_url, error)
    with make_client(transport) as client, pytest.raises(SafeTransportError) as raised:
        client.fetch(SHU_SOURCE, detail_url, purpose="detail")
    assert (raised.value.stage, raised.value.kind) == ("detail", kind)
    assert "private@example.invalid" not in str(raised.value)
    assert "proxy.invalid" not in str(raised.value)
    assert transport.request_count(detail_url) == 1 + retry_count
    assert client.request_breakdown(SHU_SOURCE.id) == {
        "robots": 1, "list": 0, "detail": 1, "retry": retry_count,
    }


@pytest.mark.parametrize("stage", ["robots", "list", "detail"])
@pytest.mark.parametrize(("error_type", "kind", "retries"), [
    (httpx.RemoteProtocolError, "remote_protocol", 0),
    (httpx.LocalProtocolError, "local_protocol", 0),
    (httpx.WriteError, "write", 0),
    (httpx.WriteTimeout, "write_timeout", 1),
    (httpx.PoolTimeout, "pool_timeout", 1),
    (httpx.CloseError, "close", 0),
    (httpx.UnsupportedProtocol, "unsupported_protocol", 0),
    (httpx.TransportError, "other", 0),
    (httpx.NetworkError, "other", 0),
    (httpx.ProtocolError, "other", 0),
    (httpx.TimeoutException, "other", 1),
])
def test_transport_subclasses_keep_safe_stage_kind_and_existing_attempt_rules(
    stage, error_type, kind, retries, caplog,
):
    from shu_insight.ingest.http_client import SafeTransportError

    secret = "synthetic-transport-secret private@example.invalid http://user:password@proxy.invalid"
    transport = RecordingTransport()
    clock = Clock()
    target = (f"{SHU_SOURCE.base_url}robots.txt" if stage == "robots"
              else SHU_SOURCE.list_url if stage == "list"
              else "https://www.shu.edu.cn/info/1/100.htm")
    if stage != "robots":
        allow_robots(transport, SHU_SOURCE)
    attempts = 1 if stage == "robots" else 1 + retries
    for _ in range(attempts):
        transport.add_exception(target, error_type(secret))

    with make_client(transport, clock=clock) as client, pytest.raises(SafeTransportError) as raised:
        client.fetch(SHU_SOURCE, target if stage != "robots" else SHU_SOURCE.list_url,
                     purpose="list" if stage != "detail" else "detail")

    assert str(raised.value) == f"transport_{stage}_{kind}"
    assert (raised.value.stage, raised.value.kind) == (stage, kind)
    assert raised.value.__context__ is None and raised.value.__cause__ is None
    assert "synthetic-transport-secret" not in repr(raised.value) + caplog.text
    expected = {"robots": 1, "list": 0, "detail": 0, "retry": 0}
    if stage != "robots":
        expected[stage] = 1
        expected["retry"] = retries
    assert client.request_breakdown(SHU_SOURCE.id) == expected
    assert client.request_count(SHU_SOURCE.id) == len(transport.requests) == sum(expected.values())
    assert transport.request_count(target) == attempts
    assert clock.sleeps == [2.0] * (sum(expected.values()) - 1)


@pytest.mark.parametrize("error_type", [httpx.WriteTimeout, httpx.PoolTimeout])
def test_new_timeout_retry_cannot_exceed_forty_physical_requests(error_type):
    transport = RecordingTransport()
    clock = Clock()
    allow_robots(transport, SHU_SOURCE)
    for _ in range(38):
        transport.add(SHU_SOURCE.list_url, body="ok")
    transport.add_exception(SHU_SOURCE.list_url, error_type("synthetic-transport-secret"))
    with make_client(transport, clock=clock) as client:
        for _ in range(38):
            client.fetch(SHU_SOURCE, SHU_SOURCE.list_url, purpose="list")
        with pytest.raises(RequestBudgetExceeded, match="^request_budget_exhausted$"):
            client.fetch(SHU_SOURCE, SHU_SOURCE.list_url, purpose="list")
        assert client.request_count(SHU_SOURCE.id) == len(transport.requests) == 40
        assert client.request_breakdown(SHU_SOURCE.id) == {
            "robots": 1, "list": 39, "detail": 0, "retry": 0,
        }
        assert clock.sleeps == [2.0] * 39
