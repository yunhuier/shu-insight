import httpx

from shu_insight.ingest.http_client import PoliteHttpClient
from shu_insight.services.notice_fulltext import prepare_fulltext
from shu_insight.services.notice_qa import Evidence

ANCHOR = Evidence("article", "比赛报名通知", "https://www.shu.edu.cn/info/1051/1001.htm",
                  "标题：比赛报名通知\n摘要：学校举办比赛，详情请查原文。", "已收录摘要")


def factory(handler):
    return lambda: PoliteHttpClient(contact="", transport=httpx.MockTransport(handler),
                                   sleep=lambda _: None)


def test_fulltext_reads_beyond_summary_and_reuses_text_cache_for_followups():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, headers={"Content-Type": "text/html; charset=utf-8"}, text=(
            '<h1>比赛报名通知</h1><div class="v_news_content"><p>'
            + "欢迎同学们参加本次比赛，积极锻炼团队合作能力。" * 100
            + "</p><p>申请材料必须包含参赛计划书和在读证明。请于指定平台在线提交。</p></div>"))

    cache = {}
    result = prepare_fulltext([ANCHOR], "在读证明", cache, client_factory=factory(handler))
    assert result.fulltext_count == 1 and not result.warnings
    assert "在读证明" in "".join(item.text for item in result.evidence)
    assert all("官网全文片段" in item.kind for item in result.evidence)
    count = len(calls)
    repeated = prepare_fulltext([ANCHOR], "计划书", cache, client_factory=factory(handler))
    assert repeated.cached_count == 1 and len(calls) == count


def test_unavailable_page_is_honest_summary_fallback():
    def handler(request):
        return httpx.Response(404 if request.url.path == "/robots.txt" else 403)

    result = prepare_fulltext([ANCHOR], "申请材料", {}, client_factory=factory(handler))
    assert result.fulltext_count == 0 and result.evidence == (ANCHOR,)
    assert "使用已收录摘要或标题" in result.warnings[0]


def test_article_redirect_cannot_fetch_another_host():
    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(302, headers={"Location": "https://example.invalid/private"})

    result = prepare_fulltext([ANCHOR], "申请材料", {}, client_factory=factory(handler))
    assert result.fulltext_count == 0 and set(hosts) == {"www.shu.edu.cn"}
