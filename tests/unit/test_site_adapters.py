from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from shu_insight.compliance.detector import scan_sensitive_text
from shu_insight.domain.models import FetchedPage, ParseState
from shu_insight.sources.base import AttachmentOnly, SourceAdapter, StructureChanged
from shu_insight.sources.registry import load_source_registry

SITE_IDS = ("shu_main", "undergraduate", "graduate", "computer")
FIXTURE_DIR = Path("tests/fixtures/sites")


@pytest.fixture
def registry():
    return load_source_registry(Path("config/sources.toml"))


@pytest.fixture
def fixture_html() -> dict[str, str]:
    return {
        source_id: (FIXTURE_DIR / f"{source_id}_list.html").read_text(encoding="utf-8")
        for source_id in SITE_IDS
    }


@pytest.mark.parametrize("source_id", SITE_IDS)
def test_adapter_discovers_public_topics_without_whole_notice_rejection(
    registry, fixture_html: dict[str, str], source_id: str
) -> None:
    adapter = registry[source_id]

    items = adapter.discover(fixture_html[source_id], adapter.config.list_url)

    assert len(items) == 2 and items[0].title == "关于举办校园人工智能创新赛的通知"
    assert all(item.url.startswith("https://") for item in items)
    assert all("utm_" not in item.url and "#" not in item.url for item in items)
    assert items[0].published_hint == {
        "shu_main": "2026.09.20",
        "undergraduate": "2026年9月20日",
        "graduate": "2026/09/20",
        "computer": "2026-09-20",
    }[source_id]


@pytest.mark.parametrize("source_id,column", [
    ("shu_main", 1000), ("undergraduate", 2000), ("graduate", 3000), ("computer", 4000),
])
def test_real_adapter_batch_exposes_safe_rejection_evidence_without_title(
    registry, fixture_html, source_id, column,
):
    adapter = registry[source_id]
    html = fixture_html[source_id].replace(f"/{column + 2}.htm", f"/{column + 2}.pdf")
    allowed, rejected = adapter.discover_with_rejections(
        html, adapter.config.list_url,
    )
    assert allowed == adapter.discover(html, adapter.config.list_url)
    assert rejected == {
        f"{adapter.config.base_url}info/{column}/{column + 2}.pdf": "attachment_blocked",
    }
    assert "名单" not in str(rejected)


@pytest.mark.parametrize("source_id", SITE_IDS)
def test_adapter_ignores_valid_notice_link_outside_reviewed_list(
    registry, fixture_html: dict[str, str], source_id: str,
) -> None:
    adapter = registry[source_id]
    stray = (
        '<footer><a href="/info/9000/9003.htm">'
        '关于举办校园人工智能创新赛的补充通知</a></footer>'
    )

    items = adapter.discover(
        fixture_html[source_id].replace("</body>", stray + "</body>"),
        adapter.config.list_url,
    )

    assert len(items) == 2 and all("补充通知" not in item.title for item in items)


@pytest.mark.parametrize("source_id,original,replacement", [
    ("shu_main", 'class="bt"', 'class="unreviewed-title"'),
    ("shu_main", "关于举办校园人工智能创新赛的通知", " "),
    ("computer", "<h3>", "<h4>"),
    ("computer", "关于举办校园人工智能创新赛的通知", " "),
])
def test_missing_or_blank_reviewed_title_stops_discovery(
    registry, fixture_html, source_id, original, replacement,
) -> None:
    adapter = registry[source_id]
    html = fixture_html[source_id].replace(original, replacement, 1)

    with pytest.raises(StructureChanged, match="^structure_changed$"):
        adapter.discover(html, adapter.config.list_url)


@pytest.mark.parametrize("source_id", SITE_IDS)
def test_adapter_parses_safe_detail_without_script_text(registry, source_id: str) -> None:
    adapter = registry[source_id]
    html = (FIXTURE_DIR / "notice_detail.html").read_text(encoding="utf-8")
    detail_url = f"{adapter.config.base_url}info/9000/9001.htm"
    page = FetchedPage(
        source_id=source_id,
        url=detail_url,
        final_url=detail_url,
        status_code=200,
        content_type="text/html; charset=utf-8",
        body=html,
        fetched_at=datetime(2026, 9, 21, tzinfo=UTC),
    )

    draft = adapter.parse_detail(page)

    assert draft.title == "关于举办校园人工智能创新赛的通知"
    assert "人工智能技术" in draft.normalized_text
    assert "ignoreThisPrompt" not in draft.normalized_text
    assert scan_sensitive_text(draft.normalized_text).blocked is False


def test_reviewed_article_inside_form_is_preserved(registry) -> None:
    adapter = registry["shu_main"]
    url = "https://www.shu.edu.cn/info/1051/400885.htm"
    body = "学校发布公开通知，说明活动安排和参加方式，请师生以官网原文为准。" * 3
    page = FetchedPage(
        source_id="shu_main", url=url, final_url=url, status_code=200,
        content_type="text/html",
        body=("<html><body><form><div class='nry'>"
              "<h1>关于开展校园活动的通知</h1>"
              f"<div class='v_news_content'><p>{body}</p>"
              "<script>ignoreThisPrompt()</script></div>"
              "</div></form></body></html>"),
        fetched_at=datetime(2026, 10, 17, tzinfo=UTC),
    )

    draft = adapter.parse_detail(page)

    assert draft.title == "关于开展校园活动的通知"
    assert body in draft.normalized_text
    assert "ignoreThisPrompt" not in draft.normalized_text


def test_computer_article_heading_is_read_from_reviewed_container(registry) -> None:
    adapter = registry["computer"]
    url = "https://cs.shu.edu.cn/info/1381/51981.htm"
    body = "学院发布公开通知，说明申请条件、办理流程和重要时间，请以官网原文为准。" * 3
    page = FetchedPage(
        source_id="computer", url=url, final_url=url, status_code=200,
        content_type="text/html",
        body=("<html><body><div class='art-main'>"
              "<div class='art-tit cont-tit d'><h3>学院2027年推免工作实施办法</h3></div>"
              f"<div class='v_news_content'><p>{body}</p></div>"
              "</div></body></html>"),
        fetched_at=datetime(2026, 10, 17, tzinfo=UTC),
    )

    draft = adapter.parse_detail(page)

    assert draft.title == "学院2027年推免工作实施办法"
    assert body in draft.normalized_text


def test_undergraduate_article_title_is_read_from_reviewed_span(registry) -> None:
    adapter = registry["undergraduate"]
    url = "https://bksy.shu.edu.cn/info/1015/322264.htm"
    body = "本科生院发布公开通知，介绍项目申报条件、提交材料和办理时间，请以原文为准。" * 3
    page = FetchedPage(
        source_id="undergraduate", url=url, final_url=url, status_code=200,
        content_type="text/html",
        body=("<html><body><div id='body'><form><table><tr><td>"
              "<span id='HRCMS_ctr13929_CalendarDetail_lblTitle'>"
              "关于开展双学士学位项目申报工作的通知</span>"
              "</td></tr><tr><td><div class='v_news_content'>"
              f"<p>{body}</p></div></td></tr></table></form></div></body></html>"),
        fetched_at=datetime(2026, 10, 17, tzinfo=UTC),
    )

    draft = adapter.parse_detail(page)

    assert draft.title == "关于开展双学士学位项目申报工作的通知"
    assert body in draft.normalized_text


def test_graduate_article_title_excludes_visit_metadata(registry) -> None:
    adapter = registry["graduate"]
    url = "https://gs.shu.edu.cn/info/1029/173312.htm"
    body = "研究生院发布公开通知，介绍学籍核对流程、办理时段及官方查询渠道。" * 3
    page = FetchedPage(
        source_id="graduate", url=url, final_url=url, status_code=200,
        content_type="text/html",
        body=("<html><body><form><table><tr><td class='NewsTitle'>"
              "上海大学研究生学籍自查通知"
              "<div>时间：2026年10月16日 访问次数：190</div>"
              "</td></tr><tr><td class='NewsBody'>"
              f"<div class='v_news_content'><p>{body}</p></div>"
              "</td></tr></table></form></body></html>"),
        fetched_at=datetime(2026, 10, 17, tzinfo=UTC),
    )

    draft = adapter.parse_detail(page)

    assert draft.title == "上海大学研究生学籍自查通知"
    assert body in draft.normalized_text


@pytest.mark.parametrize("source_id, url, title_markup", [
    (
        "undergraduate", "https://bksy.shu.edu.cn/info/1015/322264.htm",
        "<span id='HRCMS_ctr13929_CalendarDetail_lblTitle'>附件型本科通知</span>",
    ),
    (
        "graduate", "https://gs.shu.edu.cn/info/1029/173312.htm",
        "<td class='NewsTitle'>附件型研究生通知<div>时间：2026年10月16日</div></td>",
    ),
])
def test_iframe_only_official_notice_is_not_misrepresented_as_read_body(
    registry, source_id, url, title_markup,
) -> None:
    page = FetchedPage(
        source_id=source_id, url=url, final_url=url, status_code=200,
        content_type="text/html",
        body=("<html><body><form><div id='body'>" + title_markup
              + "<div class='v_news_content'>"
              + "<iframe src='/system/pdfjs/viewer.html'></iframe>"
              + "</div></div></form></body></html>"),
        fetched_at=datetime(2026, 10, 17, tzinfo=UTC),
    )

    with pytest.raises(AttachmentOnly, match="^attachment_blocked$"):
        registry[source_id].parse_detail(page)


@pytest.mark.parametrize("link_text", [
    "下载 PDF 正文",
    "关于项目申报范围和办理要求的详细说明请阅读附件全文。" * 3,
])
def test_pdf_link_text_alone_is_not_treated_as_html_article(registry, link_text) -> None:
    url = "https://bksy.shu.edu.cn/info/1015/322264.htm"
    page = FetchedPage(
        source_id="undergraduate", url=url, final_url=url, status_code=200,
        content_type="text/html",
        body=("<div id='body'><span id='HRCMS_ctr13929_CalendarDetail_lblTitle'>申报通知</span>"
              "<div class='v_news_content'>"
              f"<a href='/__local/notice.pdf'>{link_text}</a>"
              "</div></div>"),
        fetched_at=datetime(2026, 10, 17, tzinfo=UTC),
    )

    with pytest.raises(AttachmentOnly, match="^attachment_blocked$"):
        registry["undergraduate"].parse_detail(page)


def test_unrelated_iframe_without_html_article_is_not_called_attachment(registry) -> None:
    url = "https://bksy.shu.edu.cn/info/1015/322264.htm"
    page = FetchedPage(
        source_id="undergraduate", url=url, final_url=url, status_code=200,
        content_type="text/html",
        body=("<div id='body'><span id='HRCMS_ctr13929_CalendarDetail_lblTitle'>申报通知</span>"
              "<div class='v_news_content'>"
              "<iframe src='/calendar/widget.html'></iframe>"
              "</div></div>"),
        fetched_at=datetime(2026, 10, 17, tzinfo=UTC),
    )

    with pytest.raises(StructureChanged, match="^structure_changed$"):
        registry["undergraduate"].parse_detail(page)


def test_embedded_pdf_viewer_script_without_rendered_iframe_is_attachment(registry) -> None:
    url = "https://bksy.shu.edu.cn/info/1015/322264.htm"
    page = FetchedPage(
        source_id="undergraduate", url=url, final_url=url, status_code=200,
        content_type="text/html",
        body=("<div id='body'><span id='HRCMS_ctr13929_CalendarDetail_lblTitle'>申报通知</span>"
              "<div class='v_news_content'><script>"
              "showVsbpdfIframe('/__local/notice.pdf', '100%', '600');"
              "</script></div></div>"),
        fetched_at=datetime(2026, 10, 17, tzinfo=UTC),
    )

    with pytest.raises(AttachmentOnly, match="^attachment_blocked$"):
        registry["undergraduate"].parse_detail(page)


def test_pdf_object_fallback_text_alone_is_not_html_article(registry) -> None:
    url = "https://bksy.shu.edu.cn/info/1015/322264.htm"
    fallback = "完整通知放在附件中，请下载 PDF 阅读全部申报事项与时间安排。" * 3
    page = FetchedPage(
        source_id="undergraduate", url=url, final_url=url, status_code=200,
        content_type="text/html",
        body=("<div id='body'>"
              "<span id='HRCMS_ctr13929_CalendarDetail_lblTitle'>申报通知</span>"
              "<div class='v_news_content'>"
              f"<object data='/__local/notice.pdf'><p>{fallback}</p></object>"
              "</div></div>"),
        fetched_at=datetime(2026, 10, 17, tzinfo=UTC),
    )

    with pytest.raises(AttachmentOnly, match="^attachment_blocked$"):
        registry["undergraduate"].parse_detail(page)


def test_embedded_attachment_does_not_hide_substantive_html_body(registry) -> None:
    url = "https://bksy.shu.edu.cn/info/1015/322264.htm"
    body = "本科生院公开正文说明申报对象、办理时间和提交材料，请以官网原文为准。" * 3
    page = FetchedPage(
        source_id="undergraduate", url=url, final_url=url, status_code=200,
        content_type="text/html",
        body=("<form><div id='body'>"
              "<span id='HRCMS_ctr13929_CalendarDetail_lblTitle'>项目申报通知</span>"
              f"<div class='v_news_content'><p>{body}</p>"
              "<iframe src='/system/pdfjs/viewer.html'></iframe></div>"
              "</div></form>"),
        fetched_at=datetime(2026, 10, 17, tzinfo=UTC),
    )

    draft = registry["undergraduate"].parse_detail(page)

    assert draft.normalized_text == body


def test_missing_reviewed_content_container_is_typed_structure_change(registry) -> None:
    adapter = registry["computer"]
    detail_url = "https://cs.shu.edu.cn/info/9000/9002.htm"
    page = FetchedPage(
        source_id="computer",
        url=detail_url,
        final_url=detail_url,
        status_code=200,
        content_type="text/html",
        body="<html><body><h1>页面结构已改变</h1><p>短内容</p></body></html>",
        fetched_at=datetime(2026, 9, 21, tzinfo=UTC),
    )

    with pytest.raises(StructureChanged) as exc_info:
        adapter.parse_detail(page)

    assert exc_info.value.parse_state is ParseState.STRUCTURE_CHANGED


@pytest.mark.parametrize("body,expected", [
    ("请各学院按照通知要求及时提交申请材料并核对。",
     "请各学院按照通知要求及时提交申请材料并核对。"),
    ("<p>张三同学获得国家奖学金。</p>", ""),
])
def test_reviewed_short_or_personal_only_article_still_counts_as_body_read(
    registry, body, expected,
):
    url = "https://www.shu.edu.cn/info/1051/400885.htm"
    page = FetchedPage("shu_main", url, url, 200, "text/html",
                       "<h1>奖学金申请咨询13812345678 学号2312345678</h1>"
                       f"<div class='v_news_content'>{body}</div>",
                       datetime(2026, 10, 19, tzinfo=UTC))
    draft = registry["shu_main"].parse_detail(page)
    assert draft.normalized_text == expected
    assert "13812345678" not in draft.title and "2312345678" not in draft.title


@pytest.mark.parametrize("body", ["", "短内容", "<p>123456</p>"])
def test_empty_or_invalid_reviewed_container_is_not_body_read(registry, body):
    url = "https://www.shu.edu.cn/info/1051/400885.htm"
    page = FetchedPage("shu_main", url, url, 200, "text/html",
                       f"<h1>普通通知</h1><div class='v_news_content'>{body}</div>",
                       datetime(2026, 10, 19, tzinfo=UTC))
    with pytest.raises(StructureChanged):
        registry["shu_main"].parse_detail(page)


def test_long_unreviewed_content_class_is_not_an_article_container(registry) -> None:
    adapter = registry["graduate"]
    url = "https://gs.shu.edu.cn/info/1029/173242.htm"
    page = FetchedPage(
        source_id="graduate", url=url, final_url=url, status_code=200,
        content_type="text/html",
        body="<h1>普通页面</h1><div class='content'>" + "此处是网页模板内容。" * 12 + "</div>",
        fetched_at=datetime(2026, 9, 25, tzinfo=UTC),
    )
    with pytest.raises(StructureChanged):
        adapter.parse_detail(page)


@pytest.mark.parametrize("source_id,next_path", [
    ("shu_main", "/tzgg/124.htm"),
    ("undergraduate", "/index/tzgg/215.htm"),
    ("graduate", "/xwlb/sy/7.htm"),
    ("computer", "/xxgk/tzgg/3.htm"),
])
def test_official_pagination_selects_only_exact_next_text(registry, source_id, next_path):
    adapter = registry[source_id]
    html = (f'<a href="/info/1.htm">下页通知</a>'
            f'<a href="{next_path}"> 下页 </a>'
            f'<a href="/info/2.htm">往下页</a>')

    assert adapter.next_list_url(html, adapter.config.list_url) == (
        adapter.config.base_url.rstrip("/") + next_path
    )


@pytest.mark.parametrize("source_id,current_path,next_file", [
    ("shu_main", "/tzgg/124.htm", "123.htm"),
    ("undergraduate", "/index/tzgg/215.htm", "214.htm"),
    ("graduate", "/xwlb/sy/7.htm", "6.htm"),
    ("computer", "/xxgk/tzgg/3.htm", "2.htm"),
])
def test_official_numbered_page_follows_only_its_own_link(
    registry, source_id, current_path, next_file,
):
    adapter = registry[source_id]
    current = adapter.config.base_url.rstrip("/") + current_path
    assert adapter.next_list_url("<a href='/info/1.htm'>通知</a>", current) is None
    assert adapter.next_list_url(
        f'<a href="{next_file}">下页</a>', current,
    ) == current.rsplit("/", 1)[0] + "/" + next_file


@pytest.mark.parametrize("source_id", SITE_IDS)
def test_duplicate_official_next_text_is_ambiguous(registry, source_id):
    adapter = registry[source_id]
    html = '<a href="1.htm">下页</a><a href="2.htm">下页</a>'
    with pytest.raises(StructureChanged, match="^pagination_failed$"):
        adapter.next_list_url(html, adapter.config.list_url)


@pytest.mark.parametrize("source_id", SITE_IDS)
def test_official_next_text_must_lead_to_allowed_list_path(registry, source_id):
    adapter = registry[source_id]
    with pytest.raises(StructureChanged, match="^pagination_failed$"):
        adapter.next_list_url(
            '<a href="/info/1.htm">下页</a>', adapter.config.list_url,
        )


@pytest.fixture
def synthetic_pagination_adapter(registry):
    # This synthetic contract is deliberately never written to the official registry.
    return SourceAdapter(replace(
        registry["shu_main"].config,
        pagination_selector="nav.pages > a.next",
        pagination_url_pattern=r"/tzgg(?:/\d+)?\.htm",
        pagination_reviewed_on=date(2026, 10, 6),
        pagination_link_text=None,
    ))


def test_reviewed_pagination_normalizes_only_selected_next_link(synthetic_pagination_adapter):
    html = '''<main><a class="next" href="/tzgg/3.htm">正文</a></main>
    <nav class="pages"><a class="next" href="tzgg/2.htm?utm_source=x#top">下一页</a></nav>
    <footer><a class="next" href="/tzgg/4.htm">页脚</a></footer>'''
    assert synthetic_pagination_adapter.next_list_url(
        html, "https://www.shu.edu.cn/tzgg.htm"
    ) == "https://www.shu.edu.cn/tzgg/2.htm"
    assert synthetic_pagination_adapter.pagination_coverage_code == ""


def test_no_selected_next_link_does_not_infer_page(synthetic_pagination_adapter):
    html = '<footer><a class="next" href="/tzgg/2.htm">下一页</a></footer>'
    assert synthetic_pagination_adapter.next_list_url(
        html, "https://www.shu.edu.cn/tzgg.htm"
    ) is None


@pytest.mark.parametrize("href", [
    "/tzgg.htm#top", "/info/9000/1.htm", "https://other.example/tzgg/2.htm",
    "http://www.shu.edu.cn/tzgg/2.htm", "https://www.shu.edu.cn:443/tzgg/2.htm",
    "https://www.shu.edu.cn:8443/tzgg/2.htm", "https://user@www.shu.edu.cn/tzgg/2.htm",
    "/tzgg/%2e%2e/2.htm", "/tzgg/%252f2.htm", "/tzgg/%32.htm",
    "/tzgg/../tzgg/2.htm", "/tzgg/2.htm ", "/tzgg/2.htm?x=1",
    "https://www.shu.edu.cn:bad/tzgg/2.htm", "javascript:alert(1)", "",
])
def test_unsafe_or_nonlisting_next_link_is_rejected(synthetic_pagination_adapter, href):
    html = f'<nav class="pages"><a class="next" href="{href}">下一页</a></nav>'
    with pytest.raises(StructureChanged, match="^pagination_failed$"):
        synthetic_pagination_adapter.next_list_url(html, "https://www.shu.edu.cn/tzgg.htm")


@pytest.mark.parametrize("second_href", ["/tzgg/2.htm", "/tzgg/3.htm", "/info/1.htm"])
def test_duplicate_or_ambiguous_next_links_are_rejected(
    synthetic_pagination_adapter, second_href,
):
    html = ('<nav class="pages"><a class="next" href="/tzgg/2.htm">下一页</a>'
            f'<a class="next" href="{second_href}">下一页</a></nav>')
    with pytest.raises(StructureChanged, match="^pagination_failed$"):
        synthetic_pagination_adapter.next_list_url(html, "https://www.shu.edu.cn/tzgg.htm")


def test_stateless_next_link_rejects_current_page(synthetic_pagination_adapter):
    html = '<nav class="pages"><a class="next" href="2.htm">下一页</a></nav>'
    with pytest.raises(StructureChanged, match="^pagination_failed$"):
        synthetic_pagination_adapter.next_list_url(html, "https://www.shu.edu.cn/tzgg/2.htm")


def test_encoded_path_is_rejected_even_with_permissive_pattern(synthetic_pagination_adapter):
    adapter = SourceAdapter(replace(
        synthetic_pagination_adapter.config, pagination_url_pattern=r"/tzgg.*",
    ))
    html = '<nav class="pages"><a class="next" href="/tzgg/%32.htm">下一页</a></nav>'
    with pytest.raises(StructureChanged, match="^pagination_failed$"):
        adapter.next_list_url(html, "https://www.shu.edu.cn/tzgg.htm")
