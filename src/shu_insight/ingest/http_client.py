from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from shu_insight.compliance.detector import scan_sensitive_text
from shu_insight.compliance.policy import evaluate_request, is_login_or_captcha_page
from shu_insight.domain.models import FetchedPage, SourceConfig

HARD_STOP_STATUS = {401, 403, 429}
REDIRECT_STATUS = {301, 302, 303, 307, 308}
MAX_REQUESTS_PER_SOURCE = 40
_ASCII_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
ROBOTS_FAILURE_CODES = frozenset({
    "robots_not_found", "robots_forbidden", "robots_rate_limited",
    "robots_other_4xx", "robots_server_error", "robots_redirect",
    "robots_unexpected_status", "robots_wrong_content_type",
    "robots_html_or_login", "robots_missing_directives",
})
TransportKind = Literal[
    "proxy", "connect", "connect_timeout", "read", "read_timeout",
    "remote_protocol", "local_protocol", "write", "write_timeout", "pool_timeout",
    "close", "unsupported_protocol", "other",
]


def default_headers(contact: str) -> dict[str, str]:
    user_agent = "SHU-Insight-CourseProject/0.1 (+local-demo"
    candidate = contact.strip()
    if len(candidate) <= 254 and _ASCII_EMAIL.fullmatch(candidate):
        user_agent += f"; contact={candidate}"
    user_agent += ")"
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml",
    }


class SourceCircuitOpen(RuntimeError):
    pass


class RobotsDisallowed(RuntimeError):
    pass


class RobotsUnavailable(RuntimeError):
    def __init__(self, reason_code: str = "robots_unavailable") -> None:
        self.reason_code = (reason_code if reason_code in ROBOTS_FAILURE_CODES
                            else "robots_unavailable")
        super().__init__(self.reason_code)


class RedirectNotAllowed(RuntimeError):
    pass


class UnexpectedNotModified(RuntimeError):
    pass


class RequestBudgetExceeded(RuntimeError):
    reason_code = "request_budget_exhausted"

    def __init__(self) -> None:
        super().__init__(self.reason_code)


class SafeTransportError(RuntimeError):
    """Only a fixed request stage and transport category may leave this boundary."""

    def __init__(
        self,
        stage: Literal["robots", "list", "detail"],
        kind: TransportKind,
    ) -> None:
        if stage not in {"robots", "list", "detail"} or kind not in {
            "proxy", "connect", "connect_timeout", "read", "read_timeout", "other",
            "remote_protocol", "local_protocol", "write", "write_timeout", "pool_timeout",
            "close", "unsupported_protocol",
        }:
            raise ValueError("invalid_transport_category")
        self.stage = stage
        self.kind = kind
        super().__init__(f"transport_{stage}_{kind}")


def _transport_kind(
    error: httpx.TransportError,
) -> TransportKind:
    if isinstance(error, httpx.ProxyError):
        return "proxy"
    if isinstance(error, httpx.ConnectTimeout):
        return "connect_timeout"
    if isinstance(error, httpx.ReadTimeout):
        return "read_timeout"
    if isinstance(error, httpx.WriteTimeout):
        return "write_timeout"
    if isinstance(error, httpx.PoolTimeout):
        return "pool_timeout"
    if isinstance(error, httpx.ConnectError):
        return "connect"
    if isinstance(error, httpx.ReadError):
        return "read"
    if isinstance(error, httpx.WriteError):
        return "write"
    if isinstance(error, httpx.CloseError):
        return "close"
    if isinstance(error, httpx.RemoteProtocolError):
        return "remote_protocol"
    if isinstance(error, httpx.LocalProtocolError):
        return "local_protocol"
    if isinstance(error, httpx.UnsupportedProtocol):
        return "unsupported_protocol"
    return "other"


@dataclass(frozen=True, slots=True)
class CacheEntry:
    etag: str | None
    last_modified: str | None


class HttpCache:
    def __init__(self, path: Path | None = None) -> None:
        self._entries: dict[str, CacheEntry] = {}
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE IF NOT EXISTS validators "
                                   "(url_hash TEXT PRIMARY KEY, etag TEXT, last_modified TEXT)")

    def get(self, url: str) -> CacheEntry | None:
        entry = self._entries.get(url)
        if entry is None and self.path is not None:
            with sqlite3.connect(self.path) as connection:
                row = connection.execute("SELECT etag, last_modified FROM validators "
                                         "WHERE url_hash = ?",
                                         (hashlib.sha256(url.encode()).hexdigest(),)).fetchone()
            if row is not None:
                return CacheEntry(row[0], row[1])
        return entry

    def save(
        self,
        url: str,
        *,
        etag: str | None,
        last_modified: str | None,
        body: str = "",
    ) -> None:
        del body
        self._entries[url] = CacheEntry(etag, last_modified)
        if self.path is not None:
            def safe(value):
                return value if value and len(value) <= 512 and not scan_sensitive_text(
                    value).blocked else None
            with sqlite3.connect(self.path) as connection:
                connection.execute("INSERT OR REPLACE INTO validators VALUES (?, ?, ?)",
                                   (hashlib.sha256(url.encode()).hexdigest(), safe(etag),
                                    safe(last_modified)))


class SourceCircuitBreaker:
    def __init__(self) -> None:
        self._failures: dict[str, str] = {}

    def allow(self, source_id: str) -> bool:
        return source_id not in self._failures

    def reason(self, source_id: str) -> str | None:
        return self._failures.get(source_id)

    def record_failure(self, source_id: str, reason: str) -> None:
        self._failures[source_id] = reason

    def record_success(self, source_id: str) -> None:
        self._failures.pop(source_id, None)


class PoliteHttpClient:
    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        cache: HttpCache | None = None,
        breaker: SourceCircuitBreaker | None = None,
        contact: str,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache = cache or HttpCache()
        self.breaker = breaker or SourceCircuitBreaker()
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_request_at: dict[str, float] = {}
        self._request_counts: dict[str, int] = {}
        self._request_breakdowns: dict[str, dict[str, int]] = {}
        self._robots: dict[str, RobotFileParser] = {}
        self._robots_notes: dict[str, str] = {}
        self._client = httpx.Client(
            transport=transport,
            headers=default_headers(contact.strip()),
            timeout=httpx.Timeout(10.0),
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def request_count(self, source_id: str) -> int:
        return self._request_counts.get(source_id, 0)

    def request_breakdown(self, source_id: str) -> Mapping[str, int]:
        return {key: self._request_breakdowns.get(source_id, {}).get(key, 0)
                for key in ("robots", "list", "detail", "retry")}

    def robots_note(self, source_id: str) -> str:
        return self._robots_notes.get(source_id, "")

    def __enter__(self) -> PoliteHttpClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def fetch(
        self,
        source: SourceConfig,
        url: str,
        *,
        purpose: Literal["list", "detail"] = "detail",
        force_refresh: bool = False,
        redirect_target_guard: Callable[[str], None] | None = None,
    ) -> FetchedPage:
        """Fetch with normal policy gates; an optional guard may reject redirect targets."""
        if purpose not in {"list", "detail"}:
            raise ValueError("invalid_request_purpose")
        if not self.breaker.allow(source.id):
            reason = self.breaker.reason(source.id) or "unknown"
            raise SourceCircuitOpen(f"{source.id}: {reason}")

        headers: dict[str, str] = {}
        cached = None if force_refresh else self.cache.get(url)
        if cached is not None:
            if cached.etag:
                headers["If-None-Match"] = cached.etag
            if cached.last_modified:
                headers["If-Modified-Since"] = cached.last_modified

        current_url = url
        for redirect_number in range(6):
            decision = evaluate_request(source, current_url)
            if not decision.allowed:
                raise RedirectNotAllowed(decision.reason)
            if redirect_number and redirect_target_guard is not None:
                redirect_target_guard(current_url)
            self._ensure_robots(source, current_url)
            conditional_request = "If-None-Match" in headers or "If-Modified-Since" in headers
            try:
                response = self._request_with_one_retry(
                    source, current_url, headers=headers, purpose=purpose,
                )
            except httpx.TransportError as error:
                failure_kind = _transport_kind(error)
            else:
                failure_kind = None
            if failure_kind is not None:
                raise SafeTransportError(purpose, failure_kind)
            if response.status_code == 304 and not conditional_request:
                raise UnexpectedNotModified("unsolicited_not_modified")
            if response.status_code not in REDIRECT_STATUS:
                break
            if redirect_number == 5:
                raise RedirectNotAllowed("too_many_redirects")
            location = response.headers.get("location")
            if not location:
                raise RedirectNotAllowed("missing_location")
            current_url = urljoin(str(response.url), location)
            headers = {}
        if response.status_code in HARD_STOP_STATUS:
            self.breaker.record_failure(source.id, self._status_reason(response.status_code))
        elif self._contains_login_marker(response.text):
            self.breaker.record_failure(source.id, "login_or_captcha")
        elif response.status_code < 400:
            self.breaker.record_success(source.id)

        from_cache = response.status_code == 304
        if from_cache:
            body = ""
        else:
            body = response.text
            if response.status_code == 200:
                self.cache.save(
                    url,
                    etag=response.headers.get("etag"),
                    last_modified=response.headers.get("last-modified"),
                    body=body,
                )

        return FetchedPage(
            source_id=source.id,
            url=url,
            final_url=str(response.url),
            status_code=response.status_code,
            content_type=response.headers.get("content-type", ""),
            body=body,
            fetched_at=datetime.now(UTC),
            etag=response.headers.get("etag") or (cached.etag if cached else None),
            last_modified=response.headers.get("last-modified")
            or (cached.last_modified if cached else None),
            from_cache=from_cache,
        )

    def _ensure_robots(self, source: SourceConfig, url: str) -> None:
        host = urlsplit(source.base_url).hostname or ""
        parser = self._robots.get(host)
        if parser is None:
            robots_url = f"{source.base_url.rstrip('/')}/robots.txt"
            if not evaluate_request(source, robots_url, robots_request=True).allowed:
                raise RobotsUnavailable()
            try:
                response = self._request_once(source, robots_url, headers={}, purpose="robots")
            except httpx.TransportError as error:
                failure_kind = _transport_kind(error)
            else:
                failure_kind = None
            if failure_kind is not None:
                raise SafeTransportError("robots", failure_kind)
            status = response.status_code
            if status in {404, 410}:
                # RFC 9309 section 2.3.1.3: an unavailable file imposes no REP rules.
                # The normal URL, response and request-budget gates still apply.
                parser = RobotFileParser()
                parser.set_url(robots_url)
                parser.allow_all = True
                self._robots[host] = parser
                self._robots_notes[source.id] = "robots_not_published"
                return
            if status != 200:
                if status in {401, 403}:
                    reason = "robots_forbidden"
                elif status == 429:
                    reason = "robots_rate_limited"
                elif 400 <= status < 500:
                    reason = "robots_other_4xx"
                elif 500 <= status < 600:
                    reason = "robots_server_error"
                elif 300 <= status < 400:
                    reason = "robots_redirect"
                else:
                    reason = "robots_unexpected_status"
                self.breaker.record_failure(source.id, reason)
                raise RobotsUnavailable(reason)
            robots_text = response.text
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if (robots_text.lstrip("\ufeff \t\r\n").startswith("<")
                    or is_login_or_captcha_page(robots_text)):
                reason = "robots_html_or_login"
            elif content_type != "text/plain":
                reason = "robots_wrong_content_type"
            elif not (re.search(r"(?im)^\s*user-agent\s*:\s*\S", robots_text)
                      and re.search(r"(?im)^\s*(?:allow|disallow)\s*:", robots_text)):
                reason = "robots_missing_directives"
            else:
                reason = ""
            if reason:
                self.breaker.record_failure(source.id, reason)
                raise RobotsUnavailable(reason)
            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(robots_text.splitlines())
            self._robots[host] = parser
        if parser.allow_all:
            self._robots_notes[source.id] = "robots_not_published"
        if not parser.can_fetch(self._client.headers["User-Agent"], url):
            self.breaker.record_failure(source.id, "robots_disallowed")
            raise RobotsDisallowed(f"robots.txt disallows {url}")

    def _request_with_one_retry(
        self, source: SourceConfig, url: str, *, headers: dict[str, str], purpose: str,
    ) -> httpx.Response:
        try:
            return self._request_once(source, url, headers=headers, purpose=purpose)
        except (httpx.ConnectError, httpx.TimeoutException):
            return self._request_once(source, url, headers=headers, purpose="retry")

    def _request_once(
        self, source: SourceConfig, url: str, *, headers: dict[str, str], purpose: str,
    ) -> httpx.Response:
        count = self.request_count(source.id)
        if count >= MAX_REQUESTS_PER_SOURCE:
            raise RequestBudgetExceeded()
        host = urlsplit(url).hostname or ""
        previous = self._last_request_at.get(host)
        if previous is not None:
            elapsed = self._monotonic() - previous
            wait = source.minimum_interval_seconds - elapsed
            if wait > 0:
                self._sleep(wait)
        self._request_counts[source.id] = count + 1
        breakdown = self._request_breakdowns.setdefault(source.id, {})
        breakdown[purpose] = breakdown.get(purpose, 0) + 1
        try:
            return self._client.get(url, headers=headers)
        finally:
            self._last_request_at[host] = self._monotonic()

    @staticmethod
    def _status_reason(status: int) -> str:
        return {401: "authentication_required", 403: "forbidden", 429: "rate_limited"}[status]

    @staticmethod
    def _contains_login_marker(body: str) -> bool:
        return is_login_or_captcha_page(body)
