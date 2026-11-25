from __future__ import annotations

from shu_insight.ai.enhancer import NoticeEnhancer, NoticeFactPacket
from shu_insight.ai.provider import AIRequest, AIResponse, ProviderError


class FakeProvider:
    def __init__(self, response: str = '{"summary":"科研报名正在进行，请关注截止时间。"}'):
        self.response = response
        self.last_request: AIRequest | None = None
        self.calls = 0

    def complete(self, request: AIRequest) -> AIResponse:
        self.calls += 1
        self.last_request = request
        return AIResponse(self.response)


class FailingProvider:
    def complete(self, request: AIRequest) -> AIResponse:
        del request
        raise ProviderError("offline")


def make_fact_packet(
    *,
    title: str,
    deadline_text: str = "",
    source_text: str = "",
) -> NoticeFactPacket:
    return NoticeFactPacket(
        title=title,
        category="科研学术",
        deadline_text=deadline_text,
        location="宝山校区",
        audience=("本科生",),
        topic_tags=("科研",),
        excerpts=(source_text,) if source_text else (),
    )


def test_ai_request_never_contains_persona_or_phone() -> None:
    provider = FakeProvider()
    enhancer = NoticeEnhancer(provider)

    enhancer.summarize(
        make_fact_packet(
            title="科研报名通知",
            source_text="联系电话 13812345678",
        ),
        score_evidence={"matched_terms": ["科研"]},
    )

    assert provider.last_request is None


def test_provider_failure_returns_rule_based_summary() -> None:
    facts = make_fact_packet(title="报名通知", deadline_text="9月30日")

    result = NoticeEnhancer(FailingProvider()).summarize(facts, score_evidence={})

    assert result.source == "fallback"
    assert "9月30日" in result.text


def test_disabled_mode_never_calls_provider() -> None:
    provider = FakeProvider()
    facts = make_fact_packet(title="报名通知", deadline_text="9月30日")

    result = NoticeEnhancer(provider, enabled=False).summarize(facts, {})

    assert result.source == "fallback"
    assert provider.calls == 0


def test_malformed_or_overlong_ai_summary_falls_back() -> None:
    malformed = NoticeEnhancer(FakeProvider("not-json")).summarize(
        make_fact_packet(title="报名通知", deadline_text="9月30日"),
        {},
    )
    overlong = NoticeEnhancer(
        FakeProvider('{"summary":"第一句。第二句。第三句。"}')
    ).summarize(
        make_fact_packet(title="报名通知", deadline_text="9月30日"),
        {},
    )

    assert malformed.source == "fallback"
    assert overlong.source == "fallback"


def test_ai_keyword_expansions_are_disabled_until_user_enables_them() -> None:
    provider = FakeProvider('{"terms":["推荐免试","夏令营"]}')
    enhancer = NoticeEnhancer(provider)

    suggestions = enhancer.suggest_expansions(("保研",), consent_to_send_terms=True)

    assert [item.term for item in suggestions] == ["推荐免试", "夏令营"]
    assert all(item.enabled is False and item.source == "ai" for item in suggestions)


def test_ai_output_with_sensitive_data_is_rejected() -> None:
    provider = FakeProvider('{"summary":"请联系 13812345678 获取详情。"}')

    result = NoticeEnhancer(provider).summarize(
        make_fact_packet(title="报名通知", deadline_text="9月30日"),
        {},
    )

    assert result.source == "fallback"
    assert "13812345678" not in result.text


def test_candidate_review_provider_failure_uses_rule_candidates() -> None:
    facts = (make_fact_packet(title="科研报名通知"),)

    result = NoticeEnhancer(FailingProvider()).review_candidates(facts)

    assert result.source == "fallback"
    assert "1 条" in result.text


def test_preference_expansion_stays_local_without_consent_and_pii_never_leaves():
    provider = FakeProvider('{"terms":["科研"]}')
    enhancer = NoticeEnhancer(provider)
    assert enhancer.suggest_expansions(("保研",)) == ()
    assert provider.calls == 0
    assert enhancer.suggest_expansions(("13812345678",), consent_to_send_terms=True) == ()
    assert provider.calls == 0


def test_structured_contact_fields_are_sanitized_before_json_serialization():
    import json

    provider = FakeProvider()
    facts = NoticeFactPacket(title="科研通知", excerpts=("组员：张三、李四", "公开科研活动"))
    NoticeEnhancer(provider).summarize(facts, {"persona": "私有角色", "matched_terms": ["保研"]})
    assert provider.last_request is not None
    payload = json.loads(provider.last_request.text)
    assert "张三" not in provider.last_request.text
    assert "私有角色" not in provider.last_request.text
    assert "保研" not in provider.last_request.text
    assert payload["facts"]["title"] == "科研通知"
