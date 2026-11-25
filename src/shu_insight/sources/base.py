from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import PurePosixPath
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag

from shu_insight.compliance.detector import redact_public_title
from shu_insight.compliance.policy import evaluate_discovery, evaluate_request
from shu_insight.domain.models import (
    DiscoveredNotice,
    FetchedPage,
    NoticeDraft,
    ParseState,
    SourceConfig,
)

_FULL_DATE_PATTERN = re.compile(r"20\d{2}[-年/.]\d{1,2}[-月/.]\d{1,2}日?")
_CHINESE_CHARACTER_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_CONTENT_SELECTORS = (
    ".v_news_content",
    ".wp_articlecontent",
    ".article-content",
)
_TITLE_SELECTORS = ("h1", ".article-title", ".arti_title", ".news-title")
_ARTICLE_ATTACHMENT_SUFFIXES = frozenset({".pdf", ".doc", ".docx", ".xls", ".xlsx", ".zip"})


def _is_article_attachment_url(raw_url: str) -> bool:
    try:
        path = unquote(urlsplit(raw_url).path)
    except ValueError:
        return False
    return PurePosixPath(path.lower()).suffix in _ARTICLE_ATTACHMENT_SUFFIXES


def _is_article_attachment_embed(node: Tag) -> bool:
    location = str(node.get("src") or node.get("data") or "")
    if _is_article_attachment_url(location):
        return True
    try:
        path = urlsplit(location).path.lower()
    except ValueError:
        return False
    return node.name == "iframe" and path.endswith("/viewer.html")


class StructureChanged(RuntimeError):
    """Raised when a reviewed page no longer matches its defensive parser."""

    parse_state = ParseState.STRUCTURE_CHANGED


class AttachmentOnly(StructureChanged):
    """The reviewed article container has an embedded body, but no readable HTML text."""


class SourceAdapter:
    """Base contract shared by all reviewed source-specific parsers."""

    detail_title_selectors: tuple[str, ...] = _TITLE_SELECTORS

    def __init__(self, config: SourceConfig) -> None:
        self.config = config

    @property
    def pagination_coverage_code(self) -> str:
        """Expose first-page fallback without mutating the pure parser's state."""
        if not all((self.config.pagination_selector, self.config.pagination_url_pattern,
                    self.config.pagination_reviewed_on)):
            return "pagination_unverified"
        return ""

    def next_list_url(self, html: str, page_url: str) -> str | None:
        """Parse one reviewed next link; absent links end the reviewed range.

        Invalid selected links raise a safe error so callers do not mistake a
        parser/scope failure for a complete range. Traversal history belongs to
        the caller, not this stateless parser.
        """
        try:
            if not self.allows_list_url(page_url):
                raise ValueError
            if self.pagination_coverage_code:
                return None
            current = canonicalize_url(page_url)
            nodes = BeautifulSoup(html, "lxml").select(self.config.pagination_selector)
            if self.config.pagination_link_text is not None:
                nodes = [node for node in nodes if node.name == "a" and normalize_space(
                    node.get_text("", strip=True)
                ) == self.config.pagination_link_text]
            if not nodes:
                return None
            if len(nodes) != 1 or nodes[0].name != "a":
                raise ValueError
            href = nodes[0].get("href")
            if not isinstance(href, str) or not href or href != href.strip():
                raise ValueError
            if any(ord(char) < 33 for char in href):
                raise ValueError
            # urljoin resolves dot segments; reject them before normalization.
            raw_path = urlsplit(href).path
            if ("%" in raw_path or "." in raw_path.split("/")
                    or ".." in raw_path.split("/")):
                raise ValueError
            resolved = urljoin(page_url, href)
            # Check raw authority first: canonicalization discards credentials
            # and standard ports, which must never erase a policy rejection.
            if not self.allows_list_url(resolved):
                raise ValueError
            candidate = canonicalize_url(resolved)
            if candidate == current:
                raise ValueError
            return candidate
        except (ValueError, TypeError, re.error) as exc:
            raise StructureChanged("pagination_failed") from exc

    def allows_list_url(self, url: str) -> bool:
        """Pure scope shared by selections, redirects and returned list locations."""
        if not evaluate_request(self.config, url).allowed:
            return False
        try:
            candidate = canonicalize_url(url)
            parsed = urlsplit(candidate)
            if parsed.query or parsed.path.startswith("/info/"):
                return False
            if self.pagination_coverage_code:
                return candidate == canonicalize_url(self.config.list_url)
            return bool(re.fullmatch(self.config.pagination_url_pattern, parsed.path))
        except (ValueError, TypeError, re.error):
            return False

    def discover(self, html: str, page_url: str) -> list[DiscoveredNotice]:
        """Keep the public filtered-list contract for ordinary discovery callers."""
        return self.discover_with_rejections(html, page_url)[0]

    def discover_with_rejections(
        self, html: str, page_url: str,
    ) -> tuple[list[DiscoveredNotice], dict[str, str]]:
        """Pure reviewed-anchor batch with ephemeral URL/safe-code rejection evidence.

        Rejected titles are never returned. Callers decide rejection precedence
        across pages and buckets; ordinary discover still returns only safe items.
        """
        soup = BeautifulSoup(html, "lxml")
        results: list[DiscoveredNotice] = []
        rejected: dict[str, str] = {}
        for anchor in self.select_notice_anchors(soup):
            raw_href = anchor.get("href")
            if not isinstance(raw_href, str) or not raw_href.strip():
                continue
            title = self.title_hint(anchor)
            if not title:
                raise StructureChanged("structure_changed")
            title = redact_public_title(title)
            url = canonicalize_url(urljoin(page_url, raw_href))
            decision = evaluate_discovery(self.config, url, title)
            if decision.allowed:
                results.append(
                    DiscoveredNotice(
                        source_id=self.config.id,
                        url=url,
                        title=title,
                        published_hint=self.date_hint(anchor),
                    )
                )
            else:
                rejected[url] = decision.reason
        return dedupe_discoveries(results), rejected

    def parse_detail(self, page: FetchedPage) -> NoticeDraft:
        from shu_insight.processing.dates import parse_publication
        from shu_insight.processing.public_preview import safe_public_preview

        if page.source_id != self.config.id:
            raise ValueError(
                f"page source {page.source_id!r} does not match adapter {self.config.id!r}"
            )

        soup = BeautifulSoup(page.body, "lxml")
        content_nodes = [node for selector in _CONTENT_SELECTORS
                         if (node := soup.select_one(selector)) is not None]
        has_attachment_body = False
        for content_node in content_nodes:
            for embedded in content_node.select("iframe[src], embed[src], object[data]"):
                if _is_article_attachment_embed(embedded):
                    has_attachment_body = True
                    embedded.decompose()  # Object fallback text is not article text.
            if any("showVsbpdfIframe(" in script.get_text() for script in content_node.select(
                "script"
            )):
                has_attachment_body = True
            for link in content_node.select("a[href]"):
                if _is_article_attachment_url(str(link.get("href") or "")):
                    has_attachment_body = True
                    link.decompose()  # A long attachment label is not HTML article evidence.
        for unsafe_node in soup.select(
            "script, style, iframe, noscript, nav, header, footer, aside"
        ):
            unsafe_node.decompose()

        title = ""
        for selector in self.detail_title_selectors:
            title_node = soup.select_one(selector)
            if title_node is not None:
                title = self.detail_title_text(title_node)
                if title:
                    break
        if not title:
            raise StructureChanged("structure_changed")

        title = redact_public_title(title)
        normalized_text = ""
        valid_body = False
        for selector in _CONTENT_SELECTORS:
            content_node = soup.select_one(selector)
            if content_node is None:
                continue
            candidate = safe_public_preview(content_node)
            if len(_CHINESE_CHARACTER_PATTERN.findall(candidate)) >= 15:
                normalized_text = candidate
                valid_body = True
                break
            if not candidate and content_node.get_text("", strip=True):
                # The reviewed container was read, but personal material was removed.
                valid_body = True
        if not valid_body:
            if has_attachment_body:
                raise AttachmentOnly("attachment_blocked")
            raise StructureChanged("structure_changed")

        published_at = None
        for selector in ('meta[name="ArticlePublishTime"]', 'meta[name="PubDate"]',
                         'time[datetime]', '.publish-time', '.article-date'):
            node = soup.select_one(selector)
            if node is not None:
                published_at = parse_publication(
                    node.get("content") or node.get("datetime") or node.get_text(" ", strip=True)
                )
                if published_at:
                    break
        return NoticeDraft(
            source_id=self.config.id,
            canonical_url=canonicalize_url(page.final_url),
            title=title,
            normalized_text=normalized_text,
            published_at=published_at,
        )

    def select_notice_anchors(self, soup: BeautifulSoup) -> list[Tag]:
        raise NotImplementedError

    def title_hint(self, anchor: Tag) -> str:
        return normalize_space(anchor.get_text(" ", strip=True))

    def detail_title_text(self, node: Tag) -> str:
        return normalize_space(node.get_text(" ", strip=True))

    def date_hint(self, anchor: Tag) -> str | None:
        raise NotImplementedError


_TRACKING_QUERY_KEYS = {"fbclid", "gclid"}


def canonicalize_url(url: str) -> str:
    """Return a stable URL while retaining non-tracking query parameters."""

    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    if port and not ((scheme == "https" and port == 443) or (scheme == "http" and port == 80)):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_QUERY_KEYS
    ]
    query = urlencode(sorted(query_items))
    path = parsed.path or "/"
    return urlunsplit((scheme, netloc, path, query, ""))


def normalize_space(text: str) -> str:
    """Collapse HTML and line-break whitespace without joining distinct words."""

    return " ".join(text.split())


def dedupe_discoveries(items: Iterable[DiscoveredNotice]) -> list[DiscoveredNotice]:
    """Keep the first discovery for each canonical URL."""

    seen: set[str] = set()
    unique: list[DiscoveredNotice] = []
    for item in items:
        key = canonicalize_url(item.url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def nearest_full_date_hint(anchor: Tag, *, levels: int = 3) -> str | None:
    """Find a full date in the nearest reviewed list-row ancestors."""

    current: Tag | None = anchor
    for _ in range(levels):
        parent = current.parent if current is not None else None
        if not isinstance(parent, Tag):
            return None
        match = _FULL_DATE_PATTERN.search(normalize_space(parent.get_text(" ", strip=True)))
        if match:
            return match.group(0)
        current = parent
    return None
