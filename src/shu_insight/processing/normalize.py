from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from bs4 import BeautifulSoup, Tag

from shu_insight.sources.base import normalize_space

_DROP_SELECTORS = "script, style, nav, header, footer, aside, form, iframe, noscript"
_CONTENT_SELECTORS = (
    "main",
    "article",
    ".v_news_content",
    ".wp_articlecontent",
    ".article-content",
    ".content",
    "body",
)


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    text: str
    paragraphs: tuple[str, ...]
    title: str = ""
    published_at: datetime | None = None
    untrusted_source: bool = True


def _content_root(soup: BeautifulSoup) -> Tag | BeautifulSoup:
    for selector in _CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if node is not None:
            return node
    return soup


def normalize_html(
    html: str,
    *,
    title: str = "",
    published_at: datetime | None = None,
) -> NormalizedDocument:
    """Create stable paragraph text from untrusted allowed HTML."""

    soup = BeautifulSoup(html, "lxml")
    for node in soup.select(_DROP_SELECTORS):
        node.decompose()

    root = _content_root(soup)
    paragraphs = tuple(
        paragraph
        for raw_line in root.get_text("\n", strip=True).splitlines()
        if (paragraph := normalize_space(raw_line))
    )
    return NormalizedDocument(
        text="\n".join(paragraphs),
        paragraphs=paragraphs,
        title=normalize_space(title),
        published_at=published_at,
    )
