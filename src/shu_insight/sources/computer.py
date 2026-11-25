import re
from datetime import date

from bs4 import BeautifulSoup, Tag

from shu_insight.sources.base import SourceAdapter, normalize_space


class ComputerAdapter(SourceAdapter):
    """Adapter for the reviewed computer-college notice list."""

    detail_title_selectors = (".art-main .art-tit > h3", *SourceAdapter.detail_title_selectors)

    def select_notice_anchors(self, soup: BeautifulSoup) -> list[Tag]:
        return list(soup.select(".tzgg > ul > li > a[href]"))

    def title_hint(self, anchor: Tag) -> str:
        title = anchor.select_one(".tz-tx > h3")
        return normalize_space(title.get_text(" ", strip=True)) if title else ""

    def date_hint(self, anchor: Tag) -> str | None:
        month = anchor.select_one(".tz-d > span")
        day = anchor.select_one(".tz-d > b")
        if month is None or day is None:
            return None
        month_text = normalize_space(month.get_text(" ", strip=True))
        day_text = normalize_space(day.get_text(" ", strip=True))
        if not re.fullmatch(r"20\d{2}-\d{1,2}", month_text) or not day_text.isdigit():
            return None
        try:
            return date.fromisoformat(f"{month_text}-{int(day_text):02d}").isoformat()
        except ValueError:
            return None
