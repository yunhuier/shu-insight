from bs4 import BeautifulSoup, Tag

from shu_insight.sources.base import SourceAdapter, normalize_space


class ShuMainAdapter(SourceAdapter):
    """Adapter for the reviewed Shanghai University main-site notice list."""

    def select_notice_anchors(self, soup: BeautifulSoup) -> list[Tag]:
        return list(soup.select(".ej_main > .w1400 > .list > ul > li > a[href]"))

    def title_hint(self, anchor: Tag) -> str:
        title = anchor.select_one(".bt")
        return normalize_space(title.get_text(" ", strip=True)) if title else ""

    def date_hint(self, anchor: Tag) -> str | None:
        published = anchor.select_one(".sj")
        return normalize_space(published.get_text(" ", strip=True)) if published else None
