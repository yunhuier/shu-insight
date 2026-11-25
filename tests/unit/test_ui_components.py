from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

from streamlit.testing.v1 import AppTest

from shu_insight.domain.models import (
    DeadlineConfidence,
    MatchTier,
    NoticeRecord,
    ParseState,
    ProofStatus,
    Recommendation,
    ScoreBreakdown,
)
from shu_insight.ui.components import _date


def test_explicit_midnight_deadline_keeps_time_but_low_anchor_is_date_only() -> None:
    midnight = datetime(2026, 10, 11, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert _date(midnight, deadline_confidence=DeadlineConfidence.HIGH) == (
        "2026-10-11 00:00"
    )
    assert _date(midnight, deadline_confidence=DeadlineConfidence.MEDIUM) == (
        "2026-10-11 00:00"
    )
    assert _date(midnight, deadline_confidence=DeadlineConfidence.LOW) == "2026-10-11"


def test_card_renders_explicit_midnight_deadline_without_losing_precision() -> None:
    midnight = datetime(2026, 10, 11, tzinfo=ZoneInfo("Asia/Shanghai"))
    record = NoticeRecord(
        id="unverified-midnight-test", source_id="graduate",
        canonical_url="https://example.invalid/unverified-midnight-test",
        title="未核实测试数据", summary="仅测试时间格式。",
        published_at=None,
        deadline_at=midnight, deadline_confidence=DeadlineConfidence.HIGH,
        topic_tags=("synthetic_demo",),
    )
    def app(record):
        from shu_insight.ui.components import render_notice_card

        render_notice_card(record)

    at = AppTest.from_function(app, args=(record,)).run()
    assert not at.exception
    assert any("截止：2026-10-11 00:00" in caption.value for caption in at.caption)


def test_listing_card_labels_unread_body_unknown_date_and_omits_structured_details():
    record = NoticeRecord(
        id="listing-only", source_id="graduate",
        canonical_url="https://gs.shu.edu.cn/info/1029/173242.htm",
        title="官网列表通知", summary="不应显示的正文摘要", published_at=None,
        deadline_at=None, deadline_confidence=DeadlineConfidence.UNKNOWN,
        parse_state=ParseState.LISTING_ONLY, proof_status=ProofStatus.VERIFIED_LISTING,
        verified_at=datetime(2026, 10, 6, tzinfo=ZoneInfo("UTC")),
        education_levels=("不应显示的适用对象",),
    )

    def app(record):
        from shu_insight.ui.components import render_notice_card
        from shu_insight.ui.detail import render_detail

        render_notice_card(record)
        render_detail(record, {"duplicates": [], "possible_duplicates": [],
                               "versions": [{"created_at": "不应显示的版本"}]})

    at = AppTest.from_function(app, args=(record,)).run()
    assert not at.exception
    rendered = " ".join(str(item.value) for kind in ("caption", "markdown", "info")
                        for item in at.get(kind))
    assert "正文尚未读取，暂无摘要" in rendered
    assert "发布日期未知" in rendered
    assert "仅官网列表可见" in rendered
    assert "不应显示" not in rendered
    assert "官网更新" not in rendered and not at.expander and not at.dataframe


def test_parsed_empty_preview_card_explains_body_read_and_points_to_original():
    record = NoticeRecord(
        id="empty-preview", source_id="graduate",
        canonical_url="https://gs.shu.edu.cn/info/1029/173242.htm",
        title="奖学金结果公示", summary="", published_at=None, deadline_at=None,
        deadline_confidence=DeadlineConfidence.UNKNOWN, parse_state=ParseState.PARSED,
        proof_status=ProofStatus.VERIFIED_LIVE,
        verified_at=datetime(2026, 10, 19, tzinfo=ZoneInfo("UTC")),
    )

    def app(record):
        from shu_insight.ui.components import render_notice_card
        render_notice_card(record)

    at = AppTest.from_function(app, args=(record,)).run()
    assert not at.exception
    rendered = " ".join(str(item.value) for kind in ("caption", "markdown")
                        for item in at.get(kind))
    assert "正文已读取，摘要未生成" in rendered
    assert "具体内容请查看官网原文" in rendered
    assert "正文尚未读取" not in rendered and "仅官网列表可见" not in rendered


def test_recommend_shows_three_tiers_and_evidence_without_fake_relevance():
    # A numeric-score widget or ignored match_tier would misrepresent these cards.
    record = NoticeRecord(
        id="ranked", source_id="graduate",
        canonical_url="https://gs.shu.edu.cn/info/1029/173242.htm",
        title="课程办理通知", summary="按培养计划申请课程。", published_at=None,
        deadline_at=None, deadline_confidence=DeadlineConfidence.UNKNOWN,
        proof_status=ProofStatus.VERIFIED_SNAPSHOT,
        verified_at=datetime(2026, 10, 6, tzinfo=ZoneInfo("UTC")),
    )
    score = ScoreBreakdown(0, 0.5, 0, 0.9, 0, 0.215)
    recommendations = (
        Recommendation(record, score, ("匹配关键词：课程",), match_tier=MatchTier.DIRECT),
        Recommendation(replace(record, id="possible"), score, ("适用于硕士生",),
                       match_tier=MatchTier.POSSIBLE),
        Recommendation(replace(record, id="other"), score, (), match_tier=MatchTier.OTHER),
    )

    def app(recommendations):
        from shu_insight.ui.components import render_notice_card

        for recommendation in recommendations:
            render_notice_card(recommendation.notice, recommendation=recommendation)

    at = AppTest.from_function(app, args=(recommendations,)).run()
    assert not at.exception
    rendered = " ".join(str(item.value) for kind in ("caption", "markdown", "info")
                        for item in at.get(kind))
    assert all(label in rendered for label in ("直接相关", "可能适用", "其他通知"))
    assert "匹配关键词：课程" in rendered and "适用于硕士生" in rendered
    assert "暂无明确匹配证据" in rendered and "官网快照" in rendered
    assert not at.get("progress") and "相关度" not in rendered and "21.5%" not in rendered


def test_listing_only_recommendation_has_no_body_claim():
    # Rendering an unread body's stale summary/details would fabricate evidence.
    record = NoticeRecord(
        id="listing-recommendation", source_id="graduate",
        canonical_url="https://gs.shu.edu.cn/info/1029/173242.htm",
        title="科研项目通知", summary="不应显示的正文摘要", published_at=None,
        deadline_at=None, deadline_confidence=DeadlineConfidence.UNKNOWN,
        education_levels=("不应显示的适用对象",), categories=("不应显示的分类",),
        parse_state=ParseState.LISTING_ONLY, proof_status=ProofStatus.VERIFIED_LISTING,
        verified_at=datetime(2026, 10, 6, tzinfo=ZoneInfo("UTC")),
    )
    recommendation = Recommendation(
        record, ScoreBreakdown(1, 0, 0, 0.9, 0, 0.64), ("匹配关键词：科研",),
        match_tier=MatchTier.POSSIBLE,
    )

    def app(recommendation):
        from shu_insight.ui.components import render_notice_card
        from shu_insight.ui.detail import render_detail

        render_notice_card(recommendation.notice, recommendation=recommendation)
        render_detail(recommendation.notice)

    at = AppTest.from_function(app, args=(recommendation,)).run()
    assert not at.exception
    rendered = " ".join(str(item.value) for kind in ("caption", "markdown", "info")
                        for item in at.get(kind))
    assert "仅标题匹配，正文待读取" in rendered
    assert "可能适用" in rendered and "匹配关键词：科研" in rendered
    assert "仅官网列表可见" in rendered and "正文尚未读取，暂无摘要" in rendered
    assert "不应显示" not in rendered and "直接相关" not in rendered
    assert not at.expander and not at.dataframe and not at.get("progress")


def test_expired_recommendation_is_historical():
    from shu_insight.ranking.relevance import RelevanceResult
    from shu_insight.ranking.score import rank_notices

    record = NoticeRecord(
        id="expired-recommendation", source_id="graduate",
        canonical_url="https://gs.shu.edu.cn/info/1029/173242.htm",
        title="课程申请通知", summary="按培养计划申请。", published_at=None,
        deadline_at=datetime(2000, 1, 1, tzinfo=ZoneInfo("UTC")),
        deadline_confidence=DeadlineConfidence.HIGH,
    )
    recommendation = rank_notices(
        [record], {record.id: RelevanceResult(1, frozenset({"课程"}), "exact")},
        {"graduate": 0.9}, datetime(2026, 10, 6, tzinfo=ZoneInfo("UTC")), persona=None,
    )[0]

    def app(recommendation):
        from shu_insight.ui.components import render_notice_card

        render_notice_card(recommendation.notice, recommendation=recommendation)

    at = AppTest.from_function(app, args=(recommendation,)).run()
    assert not at.exception and recommendation.strong_reminder_eligible is False
    rendered = " ".join(str(item.value) for kind in ("caption", "markdown", "info")
                        for item in at.get(kind))
    assert "历史通知" in rendered and "直接相关" in rendered
    assert "距离截止时间" not in rendered and "强提醒" not in rendered


def test_listing_other_reason_is_not_presented_as_title_keyword_evidence():
    record = NoticeRecord(
        id="listing-without-keyword", source_id="graduate",
        canonical_url="https://gs.shu.edu.cn/info/1029/173242.htm",
        title="公开通知", summary="", published_at=None, deadline_at=None,
        deadline_confidence=DeadlineConfidence.UNKNOWN, parse_state=ParseState.LISTING_ONLY,
        proof_status=ProofStatus.VERIFIED_LISTING,
    )
    # Defensive input: a tier or a non-keyword reason alone cannot establish a title hit.
    recommendation = Recommendation(
        record, ScoreBreakdown(0, 1, 0, 0.9, 0, 0.34), ("适用于硕士生",),
        match_tier=MatchTier.POSSIBLE,
    )

    def app(recommendation):
        from shu_insight.ui.components import render_notice_card

        render_notice_card(recommendation.notice, recommendation=recommendation)

    at = AppTest.from_function(app, args=(recommendation,)).run()
    assert not at.exception
    rendered = " ".join(str(item.value) for kind in ("caption", "markdown", "info")
                        for item in at.get(kind))
    assert "可能适用" in rendered and "正文尚未读取，暂无摘要" in rendered
    assert "仅标题匹配" not in rendered


def test_manual_status_distinguishes_zero_new_partial_and_unchecked():
    def app():
        from shu_insight.ui.components import render_manual_source_status

        render_manual_source_status("已读来源", status={
            "state": "success", "saved_count": 3, "updated_count": 3,
            "list_pages_read": 1, "request_count": 6, "queued_count": 5,
            "coverage_code": "pagination_unverified",
        }, queued_count=0)
        render_manual_source_status("部分来源", status={
            "state": "partial", "message": "request_budget_exhausted",
            "saved_count": 2, "updated_count": 1, "list_pages_read": 2,
            "request_count": 20, "queued_count": 7,
        }, queued_count=4)
        render_manual_source_status("未读来源")

    at = AppTest.from_function(app).run()
    assert not at.exception
    rendered = " ".join(x.value for x in at.caption)
    assert "已读来源：已检查 · 新增 0 条 · 更新 3 条" in rendered
    assert "部分来源：部分检查 · 新增 1 条 · 更新 1 条" in rendered
    assert "未读来源：未检查" in rendered
    assert "最近尝试结束时积压：5 条" in rendered
    assert "当前未核实队列：0 条" in rendered
    assert "当前未核实队列：4 条" in rendered
    assert "请求预算已用尽" in rendered and "分页未验证" in rendered


def test_detail_parse_skips_are_not_labeled_as_listing_structure_changes():
    from shu_insight.ui.components import (
        SKIP_REASON_LABELS,
        SOURCE_STATE_LABELS,
        _source_status_reason_label,
    )

    assert SOURCE_STATE_LABELS["structure_changed"] == "列表结构变化"
    assert SKIP_REASON_LABELS["structure_changed"] == "正文页结构无法解析"
    assert _source_status_reason_label("partial", "structure_changed") == "正文页结构无法解析"
    assert _source_status_reason_label("structure_changed", "structure_changed") == "列表结构变化"


def test_partial_detail_parse_failure_renders_consistent_status() -> None:
    def app():
        from shu_insight.ui.components import render_manual_source_status

        render_manual_source_status("来源", status={
            "state": "partial", "message": "structure_changed",
            "list_pages_read": 3, "skip_reasons": {"structure_changed": 2},
        })

    at = AppTest.from_function(app).run()

    assert not at.exception
    rendered = " ".join(item.value for item in at.caption)
    assert "安全限制或结果：正文页结构无法解析" in rendered
    assert "跳过原因：正文页结构无法解析 2" in rendered
    assert "列表结构变化" not in rendered


def test_zero_page_robots_failure_never_claims_pagination_was_checked():
    def app():
        from shu_insight.ui.components import render_manual_source_status

        render_manual_source_status(
            "旧版误标来源",
            status={"state": "robots_unavailable", "message": "robots_unavailable",
                    "list_pages_read": 0, "coverage_code": "pagination_failed"},
        )
        render_manual_source_status(
            "本次未读来源",
            status={"state": "connection_failed", "message": "connection_failed",
                    "list_pages_read": 0, "coverage_code": ""},
        )

    at = AppTest.from_function(app).run()
    assert not at.exception
    rendered = " ".join(x.value for x in at.caption)
    assert "robots" in rendered
    assert "连接学校服务器失败" in rendered
    assert "浏览器能打开网页，不代表程序能核验 robots.txt" in rendered
    assert "浏览器与程序可能使用不同的代理或证书设置" in rendered
    assert "分页结构未通过核验" not in rendered
    assert "分页未验证" not in rendered


def test_transport_stage_labels_are_readable_and_never_echo_private_details():
    secret = "private@example.invalid http://proxy.invalid:8080"

    def app(secret):
        from shu_insight.ui.components import (
            render_manual_source_status,
            render_source_health_card,
        )

        render_manual_source_status("robots 来源", status={
            "state": "connection_failed", "message": "transport_robots_connect",
            "request_count": 1, "list_pages_read": 0,
        })
        render_manual_source_status("列表来源", status={
            "state": "connection_timeout", "message": "transport_list_read_timeout",
            "request_count": 3, "list_pages_read": 0,
        })
        render_manual_source_status("正文来源", status={
            "state": "partial", "message": "transport_detail_proxy",
            "request_count": 3, "list_pages_read": 1,
        })
        render_manual_source_status("不可信来源", status={
            "state": "partial", "message": f"transport_detail_proxy_{secret}",
        })
        render_source_health_card("评估来源", state="partial", message=secret,
                                  skip_reasons={secret: 1})

    at = AppTest.from_function(app, args=(secret,)).run()
    assert not at.exception
    rendered = " ".join(str(item.value) for kind in (
        "caption", "markdown", "metric", "info",
    ) for item in at.get(kind))
    assert "robots.txt 连接失败" in rendered
    assert "栏目页读取超时" in rendered
    assert "正文页代理连接失败" in rendered
    assert "浏览器与程序可能使用不同的代理或证书设置" in rendered
    assert "private@example.invalid" not in rendered and "proxy.invalid" not in rendered


def test_new_transport_categories_render_chinese_meaning_and_reject_suffixed_secrets():
    labels = {
        "remote_protocol": "远端协议错误", "local_protocol": "本地协议错误",
        "write": "写入失败", "write_timeout": "写入超时", "pool_timeout": "连接池等待超时",
        "close": "关闭连接失败", "unsupported_protocol": "不支持的协议",
    }

    def app(kinds):
        from shu_insight.ui.components import (
            render_manual_source_status,
            render_source_health_card,
        )

        for stage in ("robots", "list", "detail"):
            for kind in kinds:
                code = f"transport_{stage}_{kind}"
                render_manual_source_status("模拟来源", status={
                    "state": "partial", "message": code,
                })
                render_source_health_card("模拟来源", state="partial", message=code)
                unsafe = code + "_synthetic-transport-secret_private@example.invalid"
                render_manual_source_status("不可信来源", status={
                    "state": "partial", "message": unsafe,
                })
                render_source_health_card("不可信来源", state="partial", message=unsafe)

    at = AppTest.from_function(app, args=(tuple(labels),)).run()
    assert not at.exception
    rendered = " ".join(str(item.value) for kind in (
        "caption", "markdown", "metric", "info",
    ) for item in at.get(kind))
    for stage_label in ("robots.txt ", "栏目页", "正文页"):
        for kind_label in labels.values():
            assert "安全限制或结果：" + stage_label + kind_label in rendered
            assert "说明：" + stage_label + kind_label in rendered
    assert "synthetic-transport-secret" not in rendered
    assert "private@example.invalid" not in rendered


def test_manual_status_names_robots_404_without_claiming_notices_checked():
    def app():
        from shu_insight.ui.components import render_manual_source_status

        render_manual_source_status("来源", status={
            "state": "robots_not_found", "message": "robots_not_found",
            "request_count": 1, "list_pages_read": 0,
        })

    at = AppTest.from_function(app).run()
    assert not at.exception
    rendered = " ".join(x.value for x in at.caption)
    assert "robots.txt 返回 404/410" in rendered
    assert "本来源已停止抓取" in rendered
    assert "最近尝试读页数：0" in rendered


def test_manual_result_shows_session_only_listing_counts_after_structure_failure():
    from shu_insight.ingest.pipeline import (
        ListingDiagnostic,
        ListingLinkPattern,
        SourceUpdateResult,
        UpdateReport,
    )

    report = UpdateReport(
        by_source={"graduate": SourceUpdateResult(
            "structure_changed",
            listing_diagnostic=ListingDiagnostic(
                html_characters=321, link_count=12,
                scoped_article_link_count=7, selected_anchor_count=0,
                link_patterns=(ListingLinkPattern(
                    ancestor_signature="a ← li.news-item ← ul.public-list",
                    raw_info_marker=True, nearby_date=True, count=7,
                ),),
            ),
        )},
        events=(), state="failed",
    )

    def app(report):
        from shu_insight.ui.components import render_manual_update_result

        render_manual_update_result(report)

    at = AppTest.from_function(app, args=(report,)).run()
    assert not at.exception
    rendered = " ".join(item.value for item in at.caption)
    assert "栏目响应诊断" in rendered
    assert "仅本会话可见" in rendered
    assert "网页链接 12 个" in rendered
    assert "同站 /info/ 链接 7 个" in rendered
    assert "当前规则匹配 0 个" in rendered
    assert "li.news-item" in rendered
    assert "含 /info/" in rendered
    assert "邻近日期" in rendered


def test_successful_status_explains_unpublished_robots_without_stop_message():
    def app():
        from shu_insight.ui.components import render_manual_source_status

        render_manual_source_status("来源", status={
            "state": "success", "message": "robots_not_published",
            "request_count": 3, "list_pages_read": 1,
        })

    at = AppTest.from_function(app).run()
    rendered = " ".join(x.value for x in at.caption)
    assert not at.exception
    assert "robots.txt 未发布（404/410）" in rendered
    assert "本来源已停止抓取" not in rendered


def test_unfinished_manual_run_does_not_infer_abandoned_holder():
    def app():
        from shu_insight.ui.components import render_manual_run_status

        render_manual_run_status({"state": "interrupted", "finished_at": None,
                                  "started_at": "2026-10-07T00:00:00+00:00"})

    at = AppTest.from_function(app).run()
    assert not at.exception
    rendered = " ".join(x.value for x in at.caption)
    assert "尚无结束记录" in rendered and "其他页面仍在更新" in rendered
    assert "崩溃" not in rendered


def test_manual_ui_rejects_raw_reason_or_time_text():
    secret = "session-secret@example.invalid"
    def app(secret):
        from shu_insight.ui.components import render_manual_source_status

        render_manual_source_status("来源", status={
            "state": secret, "message": secret, "checked_at": secret,
            "coverage_code": secret, "skip_reasons": {secret: 1},
        })

    at = AppTest.from_function(app, args=(secret,)).run()
    assert not at.exception
    assert secret not in " ".join(str(x.value) for kind in ("caption", "markdown", "info")
                                  for x in at.get(kind))


def test_legacy_success_does_not_claim_a_known_completed_manual_scope():
    def app():
        from shu_insight.ui.components import render_manual_source_status

        render_manual_source_status("来源", last_complete_check={
            "run_id": None, "checked_at": "2026-10-06T00:00:00+00:00", "state": "success",
        })

    at = AppTest.from_function(app).run()
    assert not at.exception
    captions = " ".join(x.value for x in at.caption)
    assert "最近本轮范围完整处理：尚无记录" in captions
    assert "历史成功检查：2026-10-06 08:00:00" in captions
    assert "旧记录未记录范围" in captions


def test_legacy_source_status_does_not_present_default_phase_counts_as_observed():
    def app():
        from shu_insight.ui.components import render_manual_source_status

        render_manual_source_status("历史来源", status={
            "state": "partial", "saved_count": 7, "updated_count": 2,
            "request_count": 20, "list_pages_read": 3, "skipped_count": 4,
            "checked_at": "2026-10-18T00:00:00+00:00",
            "parsed_count": 0, "listing_count": 0, "discovered_count": 0,
            "request_breakdown": {"robots": 0, "list": 0, "detail": 0, "retry": 0},
        })

    at = AppTest.from_function(app).run()

    assert not at.exception
    rendered = " ".join(item.value for item in at.caption)
    assert "历史更新记录：新版本详细计数将在下次手动更新后显示" in rendered
    assert "历史来源：部分检查 · 新增 5 条 · 更新 2 条" in rendered
    assert "最近尝试读页数：3 · 物理请求数：20 · 跳过：4 条" in rendered
    assert "正文已读取 0 条" not in rendered and "仅列表可见 0 条" not in rendered
    assert "本轮发现：0 条" not in rendered and "robots 0 · 列表 0" not in rendered


def test_current_source_status_preserves_observed_zero_phase_counts():
    def app():
        from shu_insight.ui.components import render_manual_source_status

        render_manual_source_status("新记录", status={
            "state": "success", "request_count": 3, "list_pages_read": 1,
            "parsed_count": 0, "listing_count": 0, "discovered_count": 0,
            "request_breakdown": {"robots": 1, "list": 2, "detail": 0, "retry": 0},
        })

    at = AppTest.from_function(app).run()

    assert not at.exception
    rendered = " ".join(item.value for item in at.caption)
    assert "正文已读取 0 条 · 仅列表可见 0 条" in rendered
    assert "本轮发现：0 条" in rendered and "robots 1 · 列表 2 · 正文 0" in rendered
    assert "历史更新记录" not in rendered


def test_persisted_page_limit_uses_no_current_configuration_cap():
    def app():
        from shu_insight.ui.components import render_manual_source_status

        render_manual_source_status("历史来源", status={
            "state": "partial", "message": "page_limit_reached",
            "list_pages_read": 3, "request_count": 20,
            "coverage_code": "page_limit_reached",
        }, last_complete_check={
            "state": "success", "run_id": "historical-run",
            "checked_at": "2026-10-18T00:00:00+00:00",
            "coverage_code": "page_limit_reached",
        })

    at = AppTest.from_function(app).run()

    assert not at.exception
    rendered = " ".join(item.value for item in at.caption)
    assert "最近本轮范围完整处理：2026-10-18 08:00:00（已达当轮列表页上限）" in rendered
    assert "检查范围：已达本轮列表页上限" in rendered
    assert "安全限制或结果：已达本轮列表页上限" in rendered
    assert "5 个列表页" not in rendered
