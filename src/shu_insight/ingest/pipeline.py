from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup, Tag

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.compliance.policy import evaluate_discovery, evaluate_request, evaluate_response
from shu_insight.db.repositories import (
    SAFE_SKIP_REASONS,
    NoticeRepository,
    PendingCandidateRepository,
    SourceRepository,
)
from shu_insight.domain.models import DiscoveredNotice, FetchedPage, ProofStatus, Verification
from shu_insight.ingest.http_client import (
    RedirectNotAllowed,
    RequestBudgetExceeded,
    RobotsDisallowed,
    RobotsUnavailable,
    SafeTransportError,
    SourceCircuitOpen,
)
from shu_insight.processing.classify import Taxonomy, classify_notice, load_taxonomy
from shu_insight.processing.dates import parse_publication, utc_datetime
from shu_insight.processing.extract import extract_fields
from shu_insight.processing.normalize import NormalizedDocument
from shu_insight.processing.summarize import summarize_article
from shu_insight.sources.base import (
    AttachmentOnly,
    SourceAdapter,
    StructureChanged,
    canonicalize_url,
)


class PageClient(Protocol):
    def request_count(self, source_id: str) -> int: ...

    def fetch(
        self,
        source,
        url: str,
        *,
        purpose: Literal["list", "detail"] = "detail",
        force_refresh: bool = False,
        redirect_target_guard: Callable[[str], None] | None = None,
    ) -> FetchedPage: ...


@dataclass(frozen=True, slots=True)
class UpdateEvent:
    kind: str
    source_id: str
    message: str = ""
    url: str = ""


@dataclass(frozen=True, slots=True)
class ListingLinkPattern:
    """Bounded HTML structure only; never a title, URL, or page fragment."""

    ancestor_signature: str
    raw_info_marker: bool
    nearby_date: bool
    count: int


@dataclass(frozen=True, slots=True)
class ListingDiagnostic:
    """Session-only counts for a listing response that produced no notices."""

    html_characters: int
    link_count: int
    scoped_article_link_count: int
    selected_anchor_count: int
    link_patterns: tuple[ListingLinkPattern, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceUpdateResult:
    state: str
    saved_count: int = 0
    skipped_count: int = 0
    updated_count: int = 0
    message: str = ""
    skip_reasons: dict[str, int] = field(default_factory=dict)
    list_pages_read: int = 0
    request_count: int = 0
    queued_count: int = 0
    coverage_code: str = ""
    request_breakdown: dict[str, int] = field(default_factory=dict)
    discovered_count: int = 0
    parsed_count: int = 0
    listing_count: int = 0
    pending_count: int = 0
    listing_diagnostic: ListingDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class UpdateReport:
    by_source: dict[str, SourceUpdateResult]
    events: tuple[UpdateEvent, ...]
    run_id: str | None = None
    state: str = ""
    message: str = ""

    @property
    def saved_count(self) -> int:
        return sum(result.saved_count for result in self.by_source.values())


class SourceStopped(RuntimeError):
    def __init__(self, state: str, message: str, *, reason_code: str | None = None) -> None:
        super().__init__(message)
        self.state = state
        self.reason_code = reason_code or state


_SAFE_HTML_NAME = re.compile(r"[a-z][a-z0-9-]{0,23}\Z")
_SAFE_CLASS_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}\Z")


def _ancestor_signature(anchor: Tag) -> str:
    """Return a short, non-content structural path for session-only debugging."""
    nodes: list[str] = []
    current: Tag | None = anchor
    for _ in range(9):
        if current is None:
            break
        name = current.name if _SAFE_HTML_NAME.fullmatch(current.name or "") else "node"
        classes = current.get("class") or []
        safe_classes = [
            item for item in classes
            if isinstance(item, str) and _SAFE_CLASS_NAME.fullmatch(item)
            and not re.search(r"\d{4,}", item)
        ][:2]
        nodes.append(name + "".join(f".{item}" for item in safe_classes))
        current = current.parent if isinstance(current.parent, Tag) else None
    return " ← ".join(nodes)[:280]


def _diagnose_listing(adapter: SourceAdapter, html: str, page_url: str) -> ListingDiagnostic:
    soup = BeautifulSoup(html, "lxml")
    anchors = soup.select("a[href]")
    scoped_article_links = 0
    patterns: Counter[tuple[str, bool, bool]] = Counter()
    for anchor in anchors:
        href = anchor.get("href")
        if not isinstance(href, str):
            continue
        try:
            resolved = urljoin(page_url, href)
            if (urlsplit(resolved).path.startswith("/info/")
                    and evaluate_request(adapter.config, resolved).allowed):
                scoped_article_links += 1
                patterns[(
                    _ancestor_signature(anchor),
                    "/info/" in href,
                    bool(adapter.date_hint(anchor)),
                )] += 1
        except (TypeError, ValueError):
            continue
    return ListingDiagnostic(
        html_characters=len(html),
        link_count=len(anchors),
        scoped_article_link_count=scoped_article_links,
        selected_anchor_count=len(adapter.select_notice_anchors(soup)),
        link_patterns=tuple(
            ListingLinkPattern(signature, raw_info, nearby_date, count)
            for (signature, raw_info, nearby_date), count in sorted(
                patterns.items(), key=lambda item: (-item[1], item[0])
            )[:8]
        ),
    )


class IngestionPipeline:
    def __init__(
        self,
        *,
        registry: Mapping[str, SourceAdapter],
        client: PageClient,
        notices: NoticeRepository,
        sources: SourceRepository,
        taxonomy: Taxonomy | None = None,
        today: Callable[[], date] = date.today,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.registry = dict(registry)
        self.client = client
        self.notices = notices
        if self.notices.policy is None:
            self.notices.policy = OfficialNoticePolicy(
                {source_id: adapter.config for source_id, adapter in self.registry.items()}
            )
        self.sources = sources
        self.taxonomy = taxonomy or load_taxonomy(Path("config/taxonomy.toml"))
        self.today = today
        self.now = now
        self.pending = PendingCandidateRepository(
            notices.database, {key: adapter.config for key, adapter in self.registry.items()},
        )

    def run(
        self,
        source_ids: Sequence[str] | None = None,
        on_event: Callable[[UpdateEvent], None] | None = None,
        *,
        run_id: str | None = None,
    ) -> UpdateReport:
        events: list[UpdateEvent] = []

        def emit(event: UpdateEvent) -> None:
            events.append(event)
            if on_event is not None:
                on_event(event)

        selected_ids = tuple(source_ids) if source_ids is not None else tuple(self.registry)
        unknown = [source_id for source_id in selected_ids if source_id not in self.registry]
        if unknown:
            raise KeyError(f"unknown source ids: {', '.join(unknown)}")

        by_source: dict[str, SourceUpdateResult] = {}
        for source_id in selected_ids:
            adapter = self.registry[source_id]
            self._source_saved = 0
            self._source_updated = 0
            self._source_mining_needed = False
            self._source_list_pages = 0
            self._source_lists_complete = False
            self._source_coverage = ""
            self._source_listing_diagnostic: ListingDiagnostic | None = None
            self._source_rejected_urls: set[str] = set()
            self._source_withdrawn_urls: set[str] = set()
            self._source_listing_evidence: dict[
                str, tuple[DiscoveredNotice, str, datetime]
            ] = {}
            count_requests = getattr(self.client, "request_count", lambda _: 0)
            requests_before = count_requests(source_id)
            count_breakdown = getattr(self.client, "request_breakdown", lambda _: {})
            breakdown_before = count_breakdown(source_id)
            source_event_start = len(events)
            emit(UpdateEvent("source_started", source_id))
            try:
                # Only the explicit, locked button run mutates the URL-only queue.
                self.pending.sources[source_id] = adapter.config
                removed = self.pending.prune(source_id, now=self.now())
                for _ in range(removed):
                    emit(UpdateEvent("notice_skipped", source_id, "queue_removed"))
                if (self.today() - adapter.config.reviewed_on).days >= 90:
                    self._source_coverage = "config_review_required"
                    raise SourceStopped("config_review_required", "config_review_required")
                result = self._run_source(adapter, emit)
            except SourceStopped as exc:
                result = SourceUpdateResult(
                    exc.state,
                    saved_count=self._source_saved,
                    updated_count=self._source_updated,
                    message=exc.reason_code,
                )
                emit(UpdateEvent("source_failed", source_id, exc.reason_code))
            except (RequestBudgetExceeded, SourceCircuitOpen, RobotsDisallowed,
                    RobotsUnavailable, RedirectNotAllowed) as exc:
                reason = self._request_failure_code(exc)
                if reason == "request_budget_exhausted":
                    self._source_coverage = reason
                result = SourceUpdateResult(
                    reason, saved_count=self._source_saved,
                    updated_count=self._source_updated, message=reason,
                )
                emit(UpdateEvent("source_failed", source_id, reason))
            except Exception:
                result = SourceUpdateResult(
                    "error",
                    saved_count=self._source_saved,
                    updated_count=self._source_updated,
                    message="error",
                )
                emit(UpdateEvent("source_failed", source_id, "error"))
            if self._source_mining_needed:
                try:
                    self.notices.refresh_mining()
                except Exception:
                    result = SourceUpdateResult(
                        "error",
                        saved_count=self._source_saved,
                        updated_count=self._source_updated,
                        message="duplicate_mining_failed",
                    )
                    emit(UpdateEvent("source_failed", source_id, result.message))
            reason_counts = Counter(
                event.message if event.message in SAFE_SKIP_REASONS else "other"
                for event in events[source_event_start:]
                if event.source_id == source_id and event.kind == "notice_skipped"
            )
            if not result.message:
                robots_note = getattr(self.client, "robots_note", lambda _: "")(source_id)
                if robots_note == "robots_not_published":
                    result = replace(result, message=robots_note)
            # Listing evidence is flushed by _run_source's finally, so these
            # final proof tiers cannot accidentally count list cards as URL-only.
            visible = [row for row in self.notices.list_browseable() if row.source_id == source_id]
            visible_urls = {row.canonical_url for row in visible}
            listing_count = sum(row.proof_status is ProofStatus.VERIFIED_LISTING for row in visible)
            result = replace(result, skipped_count=sum(reason_counts.values()),
                             skip_reasons=dict(reason_counts),
                             list_pages_read=self._source_list_pages,
                             request_count=count_requests(source_id) - requests_before,
                             queued_count=self.pending.url_only_count(source_id, visible_urls),
                             pending_count=self.pending.count(source_id),
                             discovered_count=len(self._source_listing_evidence.keys()
                                                  - self._source_rejected_urls),
                             parsed_count=len(visible) - listing_count,
                             listing_count=listing_count,
                             request_breakdown={
                                 key: count_breakdown(source_id).get(key, 0)
                                 - breakdown_before.get(key, 0)
                                 for key in ("robots", "list", "detail", "retry")
                             },
                             coverage_code=self._source_coverage,
                             listing_diagnostic=self._source_listing_diagnostic)
            self._record_status(adapter, result, run_id=run_id)
            by_source[source_id] = result
            if result.state == "success":
                emit(UpdateEvent("source_finished", source_id))
        return UpdateReport(by_source=by_source, events=tuple(events), run_id=run_id)

    def _record_status(
        self, adapter: SourceAdapter, result: SourceUpdateResult, *, run_id: str | None = None,
    ) -> None:
        self.sources.record_status(
            {
                "source_id": adapter.config.id,
                "source_name": adapter.config.name,
                "base_url": adapter.config.base_url,
                "state": result.state,
                "message": result.message,
                "saved_count": result.saved_count,
                "updated_count": result.updated_count,
                "skipped_count": result.skipped_count,
                "skip_reasons": result.skip_reasons,
                "run_id": run_id,
                "list_pages_read": result.list_pages_read,
                "request_count": result.request_count,
                "queued_count": result.queued_count,
                "coverage_code": result.coverage_code,
                "request_breakdown": result.request_breakdown,
                "discovered_count": result.discovered_count,
                "parsed_count": result.parsed_count,
                "listing_count": result.listing_count,
                "pending_count": result.pending_count,
            }
        )

    @staticmethod
    def _request_failure_code(exc: Exception) -> str:
        if isinstance(exc, RobotsUnavailable):
            return exc.reason_code
        return {
            RequestBudgetExceeded: "request_budget_exhausted",
            SourceCircuitOpen: "circuit_open",
            RobotsDisallowed: "robots_disallowed",
            RedirectNotAllowed: "redirect_not_allowed",
        }[type(exc)]

    def _discover_list_pages(
        self,
        adapter: SourceAdapter,
        emit: Callable[[UpdateEvent], None],
    ) -> tuple[list[DiscoveredNotice], int, str]:
        """Read only the reviewed bounded range; stop before details on any failure."""
        source = adapter.config
        page_url: str | None = source.list_url
        visited: set[str] = set()
        seen: set[tuple[str, str]] = set()
        discoveries: list[DiscoveredNotice] = []

        redirect_targets: set[str] = set()

        def require_list_target(target: str) -> None:
            if not adapter.allows_list_url(target):
                self._source_coverage = "pagination_not_allowed"
                raise SourceStopped("partial", "pagination_not_allowed",
                                    reason_code="pagination_not_allowed")

        def guard_list_redirect(target: str) -> None:
            canonical = canonicalize_url(target)
            if canonical in visited or canonical in redirect_targets:
                self._source_coverage = "pagination_loop"
                raise SourceStopped("partial", "pagination_loop", reason_code="pagination_loop")
            require_list_target(target)
            redirect_targets.add(canonical)

        try:
            while page_url is not None:
                require_list_target(page_url)
                canonical_page = canonicalize_url(page_url)
                if canonical_page in visited:
                    self._source_coverage = "pagination_loop"
                    raise SourceStopped("partial", "pagination_loop", reason_code="pagination_loop")
                visited.add(canonical_page)
                redirect_targets.clear()
                # fetch rechecks robots and its physical-attempt budget for every page.
                list_page = self.client.fetch(
                    source, page_url, purpose="list", redirect_target_guard=guard_list_redirect,
                )
                if list_page.status_code == 304 and not list_page.body:
                    require_list_target(list_page.final_url)
                    redirect_targets.clear()
                    list_page = self.client.fetch(
                        source, page_url, purpose="list", force_refresh=True,
                        redirect_target_guard=guard_list_redirect,
                    )
                if list_page.status_code == 304 and not list_page.body:
                    raise SourceStopped("not_modified_without_body", "not_modified_without_body")
                decision = evaluate_response(
                    source, list_page.status_code, list_page.content_type,
                    list_page.final_url, list_page.body,
                )
                if not decision.allowed:
                    raise SourceStopped(decision.reason, decision.reason)
                require_list_target(list_page.final_url)
                final_page = canonicalize_url(list_page.final_url)
                if final_page != canonical_page and final_page in visited:
                    self._source_coverage = "pagination_loop"
                    raise SourceStopped("partial", "pagination_loop", reason_code="pagination_loop")
                visited.add(final_page)
                visited.update(redirect_targets)
                page_discoveries, rejected = adapter.discover_with_rejections(
                    list_page.body, list_page.final_url,
                )
                if not page_discoveries and not rejected:
                    self._source_listing_diagnostic = _diagnose_listing(
                        adapter, list_page.body, list_page.final_url,
                    )
                    raise SourceStopped("structure_changed", "structure_changed")
                for rejected_url, reason in rejected.items():
                    self.pending.remove(source.id, rejected_url)
                    self._withdraw_listing(rejected_url)
                    if rejected_url not in self._source_rejected_urls:
                        emit(UpdateEvent("notice_skipped", source.id, reason, rejected_url))
                    self._source_rejected_urls.add(rejected_url)
                self._source_list_pages += 1
                emit(UpdateEvent("list_page_read", source.id, url=canonical_page))
                for discovery in page_discoveries:
                    key = (source.id, canonicalize_url(discovery.url))
                    if key not in seen:
                        seen.add(key)
                        discoveries.append(discovery)
                        self._source_listing_evidence[discovery.url] = (
                            discovery, list_page.final_url, list_page.fetched_at,
                        )
                page_url = adapter.next_list_url(list_page.body, list_page.final_url)
                if adapter.pagination_coverage_code:
                    self._source_coverage = adapter.pagination_coverage_code
                    break
                if page_url is None:
                    self._source_coverage = "reviewed_range_complete"
                    break
                require_list_target(page_url)
                if canonicalize_url(page_url) in visited:
                    self._source_coverage = "pagination_loop"
                    raise SourceStopped("partial", "pagination_loop", reason_code="pagination_loop")
                if self._source_list_pages == 5:
                    self._source_coverage = "page_limit_reached"
                    break
        except Exception as exc:
            transport_state = ""
            if isinstance(exc, SourceStopped):
                reason = exc.reason_code
            elif isinstance(exc, (RequestBudgetExceeded, SourceCircuitOpen,
                                  RobotsDisallowed, RobotsUnavailable, RedirectNotAllowed)):
                reason = self._request_failure_code(exc)
            elif isinstance(exc, SafeTransportError):
                reason = str(exc)
                transport_state = ("connection_timeout" if exc.kind in {
                    "connect_timeout", "read_timeout", "write_timeout", "pool_timeout",
                } else "connection_failed")
            elif isinstance(exc, httpx.TimeoutException):
                reason = "connection_timeout"
            elif isinstance(exc, httpx.TransportError):
                reason = "connection_failed"
            elif isinstance(exc, StructureChanged):
                reason = ("pagination_failed" if str(exc) == "pagination_failed"
                          else "structure_changed")
            else:
                reason = "error"
            if reason in {"request_budget_exhausted", "pagination_failed"}:
                self._source_coverage = reason
            elif self._source_coverage not in {"pagination_loop", "pagination_not_allowed"}:
                self._source_coverage = ("list_check_incomplete" if self._source_list_pages
                                         else "")
            state = "partial" if self._source_list_pages else (transport_state or reason)
            raise SourceStopped(state, reason, reason_code=reason) from exc
        return discoveries, self._source_list_pages, self._source_coverage

    def _run_source(
        self,
        adapter: SourceAdapter,
        emit: Callable[[UpdateEvent], None],
    ) -> SourceUpdateResult:
        try:
            return self._process_source(adapter, emit)
        finally:
            # Wait for all list-page rejections and detail outcomes before
            # exposing lower-proof rows. A failed later page still leaves the
            # earlier reviewed evidence valid; URL-only backlog has no entry here.
            for url, (item, list_url, checked_at) in self._source_listing_evidence.items():
                if url in self._source_rejected_urls or url in self._source_withdrawn_urls:
                    continue
                self.notices.upsert_listing(item, list_url=list_url, checked_at=checked_at)
                if not self._source_lists_complete:
                    record = self.notices.find_by_url(url)
                    if record is not None and record.proof_status is ProofStatus.VERIFIED_LISTING:
                        inserted = self.pending.enqueue(adapter.config.id, url, now=self.now())
                        if not inserted and url not in self.pending.urls(adapter.config.id):
                            emit(UpdateEvent(
                                "notice_skipped", adapter.config.id, "queue_full", url,
                            ))

    def _withdraw_listing(self, url: str) -> None:
        self._source_withdrawn_urls.add(url)
        self.notices.withdraw_listing(url)

    def _process_source(
        self,
        adapter: SourceAdapter,
        emit: Callable[[UpdateEvent], None],
    ) -> SourceUpdateResult:
        source = adapter.config
        discoveries, _, _ = self._discover_list_pages(adapter, emit)
        self._source_lists_complete = True
        # A fresh rejection on any reviewed page takes precedence across every
        # scheduling bucket, including an earlier safe occurrence of this URL.
        discoveries = [item for item in discoveries if item.url not in self._source_rejected_urls]
        current_by_url = {item.url: item for item in discoveries}
        queued_urls = self.pending.urls(source.id)
        ready = self.pending.oldest_ready(source.id, limit=200, now=self.now())
        for _ in range(len(queued_urls) - len(ready)):
            emit(UpdateEvent("notice_skipped", source.id, "queue_paused"))
        parsed = {row.canonical_url: row for row in self.notices.list_verified()
                  if row.source_id == source.id}
        prior_listed = {
            row.canonical_url: DiscoveredNotice(source.id, row.canonical_url, row.title)
            for row in self.notices.list_browseable()
            if row.source_id == source.id and row.proof_status is ProofStatus.VERIFIED_LISTING
        }
        list_proven = prior_listed | current_by_url
        old = [list_proven[row["canonical_url"]] for row in ready
               if row["canonical_url"] not in parsed and row["canonical_url"] in list_proven]
        new = []
        for discovery in discoveries:
            decision = evaluate_discovery(source, discovery.url, discovery.title)
            if not decision.allowed:
                emit(UpdateEvent("notice_skipped", source.id, decision.reason, discovery.url))
                continue
            if discovery.url not in queued_urls and discovery.url not in parsed:
                new.append(discovery)

        # Parsed retries and current-list rechecks precede optional historical
        # maintenance. Every bucket, including snapshot proofs, shares one cap.
        ready_urls = {row["canonical_url"] for row in ready}
        now = utc_datetime(self.now())
        historical_cutoff = now - timedelta(hours=24)
        recent_cutoff = now - timedelta(days=90)
        parsed_ordered = self.notices.list_parsed_for_recheck(
            source.id, self._source_rejected_urls,
        )
        ready_parsed = [row for row in parsed_ordered if row.canonical_url in ready_urls]
        current_parsed = [row for row in parsed_ordered
                          if row.canonical_url in current_by_url
                          and row.canonical_url not in queued_urls]
        historical = [row for row in parsed_ordered
                      if row.canonical_url not in current_by_url
                      and row.canonical_url not in queued_urls
                      and utc_datetime(row.verified_at) <= historical_cutoff
                      and recent_cutoff <= utc_datetime(row.published_at or row.verified_at) <= now]
        required = old + new + [current_by_url.get(
            row.canonical_url, DiscoveredNotice(source.id, row.canonical_url, row.title),
        ) for row in ready_parsed + current_parsed]
        optional = [DiscoveredNotice(source.id, row.canonical_url, row.title) for row in historical]
        # Recent historical maintenance uses spare slots; deferring it does not
        # imply that the current reviewed-list scope was left incomplete.
        selected, omitted = (required + optional)[:32], required[32:]
        for discovery in omitted:
            if discovery.url in parsed:
                continue
            inserted = self.pending.enqueue(source.id, discovery.url, now=self.now())
            if not inserted and discovery.url not in self.pending.urls(source.id):
                emit(UpdateEvent("notice_skipped", source.id, "queue_full", discovery.url))
        attempted: set[str] = set()
        temporary_failure = False
        transport_failure_code = ""
        detail_structure_failure = False

        def process(discovery: DiscoveredNotice) -> None:
            nonlocal temporary_failure, transport_failure_code, detail_structure_failure
            try:
                self._process_detail(adapter, discovery, emit)
            except RobotsDisallowed:
                self.pending.remove(source.id, discovery.url)
                self._withdraw_listing(discovery.url)
                emit(UpdateEvent("notice_skipped", source.id, "robots_disallowed", discovery.url))
                raise
            except RequestBudgetExceeded:
                self._source_coverage = "request_budget_exhausted"
                # Include an interrupted redirect/retry and preserve existing
                # first-seen times through idempotent enqueue.
                for unprocessed in old + new:
                    if unprocessed.url in attempted:
                        continue
                    inserted = self.pending.enqueue(source.id, unprocessed.url, now=self.now())
                    if not inserted and unprocessed.url not in self.pending.urls(source.id):
                        emit(UpdateEvent("notice_skipped", source.id, "queue_full"))
                raise SourceStopped("partial", "request_budget_exhausted",
                                    reason_code="request_budget_exhausted") from None
            except (SafeTransportError, httpx.TransportError) as exc:
                attempted.add(discovery.url)
                temporary_failure = True
                if isinstance(exc, SafeTransportError) and not transport_failure_code:
                    transport_failure_code = str(exc)
                inserted = self.pending.enqueue(source.id, discovery.url, now=self.now())
                if not inserted and discovery.url not in self.pending.urls(source.id):
                    emit(UpdateEvent("notice_skipped", source.id, "queue_full", discovery.url))
                self.pending.mark_transient_failure(source.id, discovery.url, now=self.now())
                emit(UpdateEvent("notice_skipped", source.id, "temporary_failure", discovery.url))
            except AttachmentOnly:
                attempted.add(discovery.url)
                self.pending.remove(source.id, discovery.url)
                self._withdraw_listing(discovery.url)
                emit(UpdateEvent("notice_skipped", source.id, "attachment_blocked", discovery.url))
            except StructureChanged:
                attempted.add(discovery.url)
                self.pending.remove(source.id, discovery.url)
                detail_structure_failure = True
                emit(UpdateEvent("notice_skipped", source.id, "structure_changed", discovery.url))
            else:
                attempted.add(discovery.url)
                self.pending.remove(source.id, discovery.url)

        for discovery in selected:
            process(discovery)

        for discovery in omitted:
            emit(UpdateEvent("notice_skipped", source.id, "candidate_limit_reached", discovery.url))
        reason = ""
        if transport_failure_code:
            reason = transport_failure_code
        elif self._source_coverage not in {"", "reviewed_range_complete"}:
            reason = self._source_coverage
        elif omitted:
            reason = "candidate_limit_reached"
        elif temporary_failure:
            reason = "temporary_failure"
        elif detail_structure_failure:
            reason = "structure_changed"
        elif self.pending.count(source.id):
            reason = "pending_backlog"
        return SourceUpdateResult(
            "partial" if reason else "success", saved_count=self._source_saved,
            updated_count=self._source_updated, message=reason,
        )

    def _process_detail(
        self, adapter: SourceAdapter, discovery: DiscoveredNotice,
        emit: Callable[[UpdateEvent], None],
    ) -> None:
        source = adapter.config
        existing = self.notices.find_by_url(discovery.url)
        # Queue entries are revalidated under the current URL policy; the client
        # separately checks robots before every requested URL/redirect.
        request_decision = evaluate_request(source, discovery.url)
        if not request_decision.allowed:
            self._withdraw_listing(discovery.url)
            emit(UpdateEvent("notice_skipped", source.id, request_decision.reason, discovery.url))
            return
        try:
            detail_page = self.client.fetch(source, discovery.url, purpose="detail")
        except RedirectNotAllowed as exc:
            self._withdraw_listing(discovery.url)
            raise SourceStopped("redirect_not_allowed", "redirect_not_allowed") from exc
        if detail_page.status_code == 304 and not detail_page.body:
            if (not evaluate_request(source, detail_page.final_url).allowed
                    or canonicalize_url(detail_page.final_url) != discovery.url):
                self._withdraw_listing(discovery.url)
                raise SourceStopped("redirect_not_allowed", "not_modified_url_changed")
            if (existing is not None
                    and existing.proof_status is ProofStatus.VERIFIED_LIVE
                    and existing.verification_ref.startswith("live:")):
                self.notices.refresh_live_verification(discovery.url, detail_page.fetched_at)
                emit(
                    UpdateEvent(
                        "notice_skipped",
                        source.id,
                        "unchanged_detail",
                        discovery.url,
                    )
                )
                return
            try:
                detail_page = self.client.fetch(
                    source,
                    discovery.url,
                    purpose="detail",
                    force_refresh=True,
                )
            except RedirectNotAllowed as exc:
                self._withdraw_listing(discovery.url)
                raise SourceStopped("redirect_not_allowed", str(exc)) from exc
            if detail_page.status_code == 304 and not detail_page.body:
                raise SourceStopped(
                    "not_modified_without_body", "detail page has no fresh body"
                )
        if detail_page.status_code == 404:
            self._withdraw_listing(discovery.url)
            emit(UpdateEvent("notice_skipped", source.id, "not_found", discovery.url))
            return
        if detail_page.status_code >= 500:
            raise httpx.TransportError("temporary_failure")
        detail_decision = evaluate_response(
            source,
            detail_page.status_code,
            detail_page.content_type,
            detail_page.final_url,
            detail_page.body,
        )
        if not detail_decision.allowed:
            self._withdraw_listing(discovery.url)
            if detail_decision.reason in {
                "authentication_required",
                "forbidden",
                "rate_limited",
                "login_or_captcha",
            }:
                raise SourceStopped(detail_decision.reason, detail_decision.reason)
            emit(
                UpdateEvent(
                    "notice_skipped",
                    source.id,
                    detail_decision.reason,
                    discovery.url,
                )
            )
            return
        draft = adapter.parse_detail(detail_page)

        title_decision = evaluate_discovery(source, draft.canonical_url, draft.title)
        if not title_decision.allowed:
            self._withdraw_listing(discovery.url)
            emit(UpdateEvent("notice_skipped", source.id, title_decision.reason, discovery.url))
            return
        draft = replace(
            draft, canonical_url=discovery.url,
            published_at=draft.published_at or parse_publication(discovery.published_hint)
            or (existing.published_at if existing and self.notices.policy.browseable(existing)
                and existing.canonical_url == discovery.url else None),
        )
        document = NormalizedDocument(
            text=draft.normalized_text,
            paragraphs=tuple(
                part for part in draft.normalized_text.splitlines() if part.strip()
            ),
            title=draft.title,
            published_at=draft.published_at,
        )
        summary = summarize_article(document.text)
        enriched = draft
        if document.text:
            fields = extract_fields(document, detail_page.fetched_at)
            classification = classify_notice(draft.title, document.text, self.taxonomy)
            categories = classification.categories or (classification.primary,)
            enriched = replace(
                draft,
                deadline_at=fields.deadline_at,
                deadline_confidence=fields.deadline_confidence,
                location=fields.locations[0].value if fields.locations else "",
                education_levels=tuple(item.value for item in fields.education_levels),
                grades=tuple(item.value for item in fields.grades),
                colleges=tuple(item.value for item in fields.colleges),
                majors=tuple(dict.fromkeys((
                    *(item.value for item in fields.majors), *classification.major_mentions,
                ))),
                categories=categories,
                topic_tags=tuple(sorted(classification.tags)),
            )
        proof = Verification(
            ProofStatus.VERIFIED_LIVE,
            detail_page.fetched_at,
            "live:" + hashlib.sha256(detail_page.body.encode("utf-8")).hexdigest(),
        )
        upsert = self.notices.upsert(
            enriched, verification=proof, summary_override=summary,
            refresh_mining=False,
        )
        if upsert.created or upsert.updated:
            # Refresh removes stale clusters when a previously readable body
            # becomes empty; fresh personal-only rows never initiate mining.
            self._source_mining_needed |= bool(document.text or (existing and existing.summary))
            self._source_saved += 1
            self._source_updated += int(upsert.updated)
            emit(UpdateEvent("notice_saved", source.id, url=discovery.url))
        else:
            emit(UpdateEvent("notice_skipped", source.id, "content_unchanged", discovery.url))
        del detail_page
