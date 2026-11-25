import re

from bs4 import BeautifulSoup, Tag

from shu_insight.sources.base import SourceAdapter, normalize_space


class GraduateAdapter(SourceAdapter):
    """Adapter for the reviewed graduate-school notice list."""

    detail_title_selectors = ("td.NewsTitle", *SourceAdapter.detail_title_selectors)

    def detail_title_text(self, node: Tag) -> str:
        if node.name == "td" and "NewsTitle" in node.get("class", []):
            return normalize_space(" ".join(node.find_all(string=True, recursive=False)))
        return super().detail_title_text(node)

    def select_notice_anchors(self, soup: BeautifulSoup) -> list[Tag]:
        return list(soup.select('table[width="755"] tr[id^="line_u17_"] td:first-of-type a[href]'))

    def date_hint(self, anchor: Tag) -> str | None:
        row = anchor.find_parent("tr")
        cells = row.find_all("td", recursive=False) if row else []
        if len(cells) < 2:
            return None
        match = re.search(r"20\d{2}/\d{1,2}/\d{1,2}", normalize_space(cells[1].get_text()))
        return match.group(0) if match else None
