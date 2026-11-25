from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.compliance.policy import evaluate_discovery, evaluate_response
from shu_insight.domain.models import (
    DeadlineConfidence,
    NoticeRecord,
    ParseState,
    ProofStatus,
    SourceConfig,
)
from shu_insight.sources.registry import load_source_registry

CS_SOURCE = SourceConfig(
    id="computer",
    name="上海大学计算机工程与科学学院",
    base_url="https://cs.shu.edu.cn/",
    list_url="https://cs.shu.edu.cn/xxgk/tzgg.htm",
    allowed_path_prefixes=("/xxgk/tzgg", "/info/"),
    authority_score=0.95,
    minimum_interval_seconds=2.0,
    reviewed_on=date(2026, 9, 21),
)


@pytest.fixture
def official_policy() -> OfficialNoticePolicy:
    registry = load_source_registry(Path("config/sources.toml"))
    return OfficialNoticePolicy({key: item.config for key, item in registry.items()})


@pytest.mark.parametrize("url", [
    "http://cs.shu.edu.cn/info/1/2.htm",
    "https://cs.shu.edu.cn:444/info/1/2.htm",
    "https://cs.shu.edu.cn:443/info/1/2.htm",
    "https://user@cs.shu.edu.cn/info/1/2.htm",
    "https://cs.shu.edu.cn/info/%2e%2e/private.htm",
    "https://cs.shu.edu.cn/info/%2fprivate.htm",
    "https://cs.shu.edu.cn/info/%5cprivate.htm",
    "https://cs.shu.edu.cn/info/../private.htm",
    "https://cs.shu.edu.cn/info/./2.htm",
    "https://cs.shu.edu.cn/infoevil/2.htm",
    "https://demo.invalid/computer/demo-1",
])
def test_official_policy_rejects_disguised_urls(
    official_policy: OfficialNoticePolicy, url: str
) -> None:
    assert not official_policy.allows("computer", url, "讲座通知", ())


def test_official_policy_requires_exact_source_and_excludes_demo_markers(
    official_policy: OfficialNoticePolicy,
) -> None:
    url = "https://gs.shu.edu.cn/info/1029/173242.htm"
    title = "上海大学2026级硕博连读研究生、直博生 申请课程免修及成绩认定通知"
    assert official_policy.allows("graduate", url, title, ())
    assert not official_policy.allows("unknown", url, title, ())
    assert not official_policy.allows("computer", url, title, ())
    assert not official_policy.allows("graduate", url, f"[演示]{title}", ())
    assert not official_policy.allows("graduate", url, title, ("synthetic_demo",))


@pytest.mark.parametrize(("source_id", "list_url", "allowed"), [
    ("graduate", "https://gs.shu.edu.cn/xwlb/sy.htm", True),
    ("graduate", "https://gs.shu.edu.cn/xwlb/sy/12.htm", True),
    ("computer", "https://cs.shu.edu.cn/xxgk/tzgg/12.htm", True),
    ("unknown", "https://gs.shu.edu.cn/xwlb/sy.htm", False),
    ("computer", "https://gs.shu.edu.cn/xwlb/sy.htm", False),
    ("graduate", "http://gs.shu.edu.cn/xwlb/sy.htm", False),
    ("graduate", "https://gs.shu.edu.cn:443/xwlb/sy.htm", False),
    ("graduate", "https://user@gs.shu.edu.cn/xwlb/sy.htm", False),
    ("graduate", "https://gs.shu.edu.cn/info/1/2.htm", False),
    ("graduate", "https://gs.shu.edu.cn/xwlb/sy2.htm", False),
    ("graduate", "https://gs.shu.edu.cn/xwlb/sy/private.htm", False),
    ("graduate", "https://gs.shu.edu.cn/xwlb/sy/%2e%2e/12.htm", False),
    ("graduate", "https://gs.shu.edu.cn/xwlb/sy.htm?token=secret", False),
    ("graduate", "https://gs.shu.edu.cn/xwlb/sy.htm#fragment", False),
])
def test_official_listing_policy_requires_reviewed_list_path(
    official_policy, source_id, list_url, allowed,
):
    assert official_policy.allows_list_page(source_id, list_url) is allowed


def test_listing_pagination_requires_review_and_current_request_scope(official_policy):
    source = official_policy.sources["graduate"]
    url = "https://gs.shu.edu.cn/xwlb/sy/12.htm"
    unreviewed = OfficialNoticePolicy({"graduate": replace(source, pagination_reviewed_on=None)})
    narrowed = OfficialNoticePolicy({"graduate": replace(
        source, allowed_path_prefixes=("/info/",),
    )})
    assert not unreviewed.allows_list_page("graduate", url)
    assert not narrowed.allows_list_page("graduate", source.list_url)


def test_listing_browseability_does_not_claim_parsed_evidence(official_policy):
    record = NoticeRecord(
        id="listing", source_id="graduate",
        canonical_url="https://gs.shu.edu.cn/info/1029/173242.htm", title="课程免修通知",
        summary="", published_at=None, deadline_at=None,
        deadline_confidence=DeadlineConfidence.UNKNOWN,
        parse_state=ParseState.LISTING_ONLY, proof_status=ProofStatus.VERIFIED_LISTING,
        verified_at=datetime(2026, 10, 19, tzinfo=UTC), verification_ref="listing:" + "a" * 64,
    )
    assert official_policy.browseable(record)
    assert not official_policy.visible(record)
    for changed in (
        replace(record, verification_ref=""),
        replace(record, verification_ref="live:fixture"),
        replace(record, verified_at=None),
        replace(record, verified_at=datetime(2026, 10, 19)),
        replace(record, proof_status=ProofStatus.ATTACHMENT_EXCLUDED),
        replace(record, proof_status=ProofStatus.UNVERIFIED_LEGACY),
        replace(record, canonical_url="https://evil.example/info/1/2.htm"),
        replace(record, parse_state=ParseState.PARSED),
    ):
        assert not official_policy.browseable(changed)
    parsed = replace(record, parse_state=ParseState.PARSED,
                     proof_status=ProofStatus.VERIFIED_LIVE, verification_ref="live:fixture")
    assert official_policy.visible(parsed) and official_policy.browseable(parsed)
    assert not official_policy.visible(replace(parsed, parse_state=ParseState.LISTING_ONLY))


def test_official_policy_requires_complete_record_proof(
    official_policy: OfficialNoticePolicy,
) -> None:
    record = NoticeRecord(
        id="one", source_id="graduate",
        canonical_url="https://gs.shu.edu.cn/info/1029/173242.htm",
        title="上海大学2026级硕博连读研究生、直博生 申请课程免修及成绩认定通知",
        summary="2026级硕博连读研究生和直博生可按条件申请课程免修或成绩认定。",
        published_at=datetime(2026, 9, 11, 5, 50, tzinfo=UTC),
        deadline_at=datetime(2026, 10, 11, tzinfo=UTC),
        deadline_confidence=DeadlineConfidence.LOW,
        proof_status=ProofStatus.VERIFIED_SNAPSHOT,
        verified_at=datetime(2026, 9, 24, tzinfo=UTC), verification_ref="snapshot:entry-1",
    )
    assert official_policy.visible(record)
    assert record.deadline_at.date().isoformat() == "2026-10-11"
    assert record.deadline_confidence == DeadlineConfidence.LOW
    assert not official_policy.visible(replace(record, verification_ref=""))
    assert not official_policy.visible(replace(record, verified_at=None))
    assert not official_policy.visible(replace(record, verified_at=datetime(2026, 9, 24)))
    assert not official_policy.visible(replace(record, proof_status=ProofStatus.UNVERIFIED_LEGACY))


@pytest.mark.parametrize(
    "title",
    [
        "硕士研究生进入复试阶段名单",
        "学位评定委员会授予硕士学位人员公告",
        "家庭经济困难认定结果公示",
        "关于2026年研究生国家奖学金评选结果的公示",
        "拟录取名单的公示",
        "拟录取人员名单（第一批）公示",
        "关于2026年研究生国家奖学金拟获奖人选公示",
        "关于公示2026年研究生国家奖学金评选结果的通知",
        "关于张三同学记过处分的公告",
        "关于2026年度家庭经济困难学生认定结果的通知",
    ],
)
def test_personal_result_topic_does_not_deny_official_discovery(title: str) -> None:
    result = evaluate_discovery(CS_SOURCE, "https://cs.shu.edu.cn/info/1/2.htm", title)

    assert result.allowed is True
    assert result.reason == "allowed"


@pytest.mark.parametrize("title", [
    "关于推荐名单填报工作的通知",
    "关于推荐名单公示工作安排的通知",
    "关于推荐名单公布后申诉流程的通知",
    "关于办理退学手续的说明",
    "关于公布2026年度校级重点课程建设项目立项名单的通知",
    "上海大学学生纪律处分管理办法学习通知",
    "关于学习学生处分处理决定流程的通知",
    "关于2026年学生奖学金评选结果公示安排的通知",
    "关于拟录取人员名单（第一批）公示工作安排的通知",
    "关于开展家庭经济困难认定申请工作的通知",
    "关于研究生奖学金评选结果公示异议受理的通知",
    "关于退学处理决定送达程序的通知",
    "关于开展2026年研究生国家奖学金评选工作的通知",
    "关于组织2026年度研究生国家奖学金评选的通知",
    "关于开展2026年秋季学期家庭经济困难学生认定工作的通知",
    "关于报名参加课程讲座的通知（咨询电话13800138000）",
])
def test_general_notice_title_is_not_mistaken_for_personal_result(title: str) -> None:
    result = evaluate_discovery(CS_SOURCE, "https://cs.shu.edu.cn/info/1/2.htm", title)
    assert result.allowed and result.reason == "allowed"


def test_named_candidate_topic_is_admitted_for_safe_body_preview() -> None:
    result = evaluate_discovery(
        CS_SOURCE,
        "https://cs.shu.edu.cn/info/1/2.htm",
        "2026年研究生奖学金候选人公示",
    )
    assert result.allowed


@pytest.mark.parametrize("title", ["奖学金申请", "录取流程", "处分规定"])
def test_topic_words_alone_do_not_reject_official_notice(title):
    assert evaluate_discovery(CS_SOURCE, "https://cs.shu.edu.cn/info/1/2.htm", title).allowed


@pytest.mark.parametrize("title", ["", "  ", "[演示]普通通知"])
def test_discovery_still_requires_real_nonempty_title(title):
    assert not evaluate_discovery(CS_SOURCE, "https://cs.shu.edu.cn/info/1/2.htm", title).allowed


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://cs.shu.edu.cn/info/1/2.htm", "https_required"),
        ("https://evil.example/info/1/2.htm", "host_not_allowed"),
        ("https://cs.shu.edu.cn/index/news.htm", "path_not_allowed"),
        ("https://cs.shu.edu.cn/xxgk/tzgg-evil.htm", "path_not_allowed"),
        ("https://cs.shu.edu.cn/info/file.pdf", "attachment_blocked"),
    ],
)
def test_discovery_rejects_urls_outside_allowlist(url: str, reason: str) -> None:
    assert evaluate_discovery(CS_SOURCE, url, "普通通知").reason == reason


def test_safe_notice_discovery_is_allowed() -> None:
    decision = evaluate_discovery(
        CS_SOURCE,
        "https://cs.shu.edu.cn/info/1381/51981.htm",
        "关于举办校园人工智能创新赛的通知",
    )

    assert decision.allowed is True
    assert decision.reason == "allowed"


def test_numbered_column_pagination_remains_in_scope() -> None:
    decision = evaluate_discovery(
        CS_SOURCE, "https://cs.shu.edu.cn/xxgk/tzgg2.htm", "普通栏目",
    )
    assert decision.allowed is True


@pytest.mark.parametrize(
    ("status", "content_type", "url", "body", "reason"),
    [
        (403, "text/html", "https://cs.shu.edu.cn/info/1/2.htm", "", "forbidden"),
        (429, "text/html", "https://cs.shu.edu.cn/info/1/2.htm", "", "rate_limited"),
        (200, "application/pdf", "https://cs.shu.edu.cn/info/1/2.htm", "", "attachment_blocked"),
        (200, "text/plain", "https://cs.shu.edu.cn/info/1/2.htm", "", "unsupported_content_type"),
        (
            302, "text/html", "https://cs.shu.edu.cn/info/1/2.htm",
            "<h1>文章</h1>", "unresolved_redirect",
        ),
        (200, "text/html", "https://evil.example/info/1/2.htm", "", "host_not_allowed"),
        (
            200,
            "text/html",
            "https://cs.shu.edu.cn/info/1/2.htm",
            "请输入验证码",
            "login_or_captcha",
        ),
    ],
)
def test_response_gate_returns_specific_denial_reason(
    status: int, content_type: str, url: str, body: str, reason: str
) -> None:
    decision = evaluate_response(CS_SOURCE, status, content_type, url, body)

    assert decision.allowed is False
    assert decision.persistable is False
    assert decision.reason == reason


def test_safe_html_response_can_be_persisted() -> None:
    decision = evaluate_response(
        CS_SOURCE,
        200,
        "text/html; charset=utf-8",
        "https://cs.shu.edu.cn/info/1381/51981.htm",
        "关于举办校园人工智能创新赛的通知，报名截止时间为10月1日。",
    )

    assert decision.allowed is True
    assert decision.persistable is True


def test_response_gate_does_not_treat_template_footer_as_article_content() -> None:
    decision = evaluate_response(
        CS_SOURCE, 200, "text/html",
        "https://cs.shu.edu.cn/info/1/2.htm",
        "<article>官网通知正文</article><footer>info@shu.edu.cn 登录</footer>",
    )
    assert decision.allowed is True


def test_real_article_mention_of_unified_identity_is_not_auth_page() -> None:
    decision = evaluate_response(
        CS_SOURCE, 200, "text/html",
        "https://cs.shu.edu.cn/info/1/2.htm",
        "<h1>选课通知</h1><div class='v_news_content'>学生使用统一身份认证账号登录选课系统。</div>",
    )
    assert decision.allowed is True
