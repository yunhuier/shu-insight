from bs4 import BeautifulSoup, Tag

from shu_insight.sources.base import SourceAdapter, normalize_space


class UndergraduateAdapter(SourceAdapter):
    """Adapter for the reviewed undergraduate-school notice list."""

    detail_title_selectors = (
        '#body span[id$="_CalendarDetail_lblTitle"]',
        *SourceAdapter.detail_title_selectors,
    )

    def select_notice_anchors(self, soup: BeautifulSoup) -> list[Tag]:
        return list(soup.select("#body .only-listPage > .only-list > ul > li > a[href]"))

    def date_hint(self, anchor: Tag) -> str | None:
        row = anchor.find_parent("li")
        published = row.find("span", recursive=False) if row else None
        return normalize_space(published.get_text(" ", strip=True)) if published else None
