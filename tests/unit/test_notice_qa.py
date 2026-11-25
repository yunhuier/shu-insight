import json
from dataclasses import replace
from datetime import UTC, datetime

import httpx
import pytest

from shu_insight.domain.models import DeadlineConfidence, NoticeRecord, ParseState, ProofStatus


def record(number=1, title="奖学金申请通知", summary="本科生可在线申请奖学金，材料要求见原通知。"):
    return NoticeRecord(
        str(number), "shu_main", f"https://www.shu.edu.cn/info/1051/{number}.htm",
        title, summary, datetime(2026, 10, 1, tzinfo=UTC), None, DeadlineConfidence.UNKNOWN,
        proof_status=ProofStatus.VERIFIED_SNAPSHOT,
    )


def test_bm25_anchors_current_notice_and_finds_only_matching_related_records():
    from shu_insight.services.notice_qa import retrieve_evidence

    anchor = record()
    related = record(2, "奖学金申请材料说明", "奖学金申请需要在线提交材料。")
    other = record(3, "图书馆闭馆", "图书馆周末暂停开放。")
    evidence = retrieve_evidence([anchor, related, other], anchor.id, "奖学金材料怎么提交？")
    assert [item.notice_id for item in evidence] == [anchor.id, related.id]
    assert len(retrieve_evidence([anchor, related], anchor.id, "怎么申请？", related=False)) == 1


def test_context_uses_public_summary_and_never_legacy_listing_body_or_unverified_data():
    from shu_insight.services.notice_qa import retrieve_evidence

    anchor = record(summary="请在线报名。 联系人：测试甲，电话：66133507-803")
    listing = replace(record(2), parse_state=ParseState.LISTING_ONLY,
                      proof_status=ProofStatus.VERIFIED_LISTING, summary="不该出现的秘密正文")
    unverified = replace(record(3), proof_status=ProofStatus.UNVERIFIED_LEGACY)
    evidence = retrieve_evidence([anchor, listing, unverified], anchor.id, "奖学金")
    content = str(evidence)
    assert "测试甲" not in content and "66133507" not in content
    assert "秘密正文" not in content and unverified.id not in [item.notice_id for item in evidence]
    assert evidence[1].kind == "仅列表标题"


def test_related_retrieval_does_not_match_generated_metadata_labels():
    from shu_insight.services.notice_qa import retrieve_evidence

    anchor = record()
    other = record(2, "图书馆开放安排", "周末图书馆开放。")
    assert len(retrieve_evidence([anchor, other], anchor.id, "发布时间")) == 1


def test_deepseek_sends_bounded_evidence_once_with_key_only_in_header():
    from shu_insight.services.notice_qa import DeepSeekClient, retrieve_evidence

    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "answer": "已收录内容只说明可以在线申请，未提供具体材料清单。", "citations": [1],
        })}}]})

    key = "test-only-secret-not-a-real-key"
    client = DeepSeekClient(key, transport=httpx.MockTransport(handler))
    evidence = retrieve_evidence([record()], "1", "需要什么材料？")
    result = client.answer("需要什么材料？", evidence, history=[])
    assert len(requests) == 1 and "材料" in result.answer
    payload = json.loads(requests[0].content)
    assert requests[0].url == "https://api.deepseek.com/chat/completions"
    assert requests[0].headers["Authorization"] == f"Bearer {key}"
    assert key not in requests[0].content.decode() and key not in repr(client)
    assert payload["model"] == "deepseek-flash" and payload["stream"] is False
    assert result.sources == (evidence[0],)


@pytest.mark.parametrize("status", [401, 402, 429, 500])
def test_provider_failures_do_not_leak_private_exception_text_or_retry(status):
    from shu_insight.services.notice_qa import DeepSeekClient, NoticeQaError, retrieve_evidence

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="unsafe-provider-body-secret")

    client = DeepSeekClient("test-key", transport=httpx.MockTransport(handler))
    with pytest.raises(NoticeQaError) as error:
        client.answer("如何申请", retrieve_evidence([record()], "1", "申请"), [])
    assert "unsafe-provider" not in str(error.value) and len(calls) == 1


@pytest.mark.parametrize("citations", [[2], [True], [], ["https://example.invalid"]])
def test_invalid_citations_cannot_create_fabricated_sources(citations):
    from shu_insight.services.notice_qa import DeepSeekClient, NoticeQaError, retrieve_evidence

    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={
        "choices": [{"message": {"content": json.dumps({
            "answer": "回答", "citations": citations,
        })}}],
    }))
    with pytest.raises(NoticeQaError):
        DeepSeekClient("test-key", transport=transport).answer(
            "如何申请", retrieve_evidence([record()], "1", "申请"), [])


def test_empty_question_missing_key_or_context_never_calls_provider():
    from shu_insight.services.notice_qa import DeepSeekClient, NoticeQaError

    def forbidden(_):
        pytest.fail("provider must not be called")

    for key, question in [("", "问题"), ("test-key", " "), ("test-key", "问题")]:
        with pytest.raises(NoticeQaError):
            DeepSeekClient(key, transport=httpx.MockTransport(forbidden)).answer(question, [], [])
