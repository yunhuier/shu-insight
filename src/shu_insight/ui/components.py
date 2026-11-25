from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta
from typing import Any

import streamlit as st

from shu_insight.domain.models import (
    DeadlineConfidence,
    MatchTier,
    NoticeRecord,
    ParseState,
    ProofStatus,
    Recommendation,
)
from shu_insight.ingest.pipeline import UpdateEvent, UpdateReport
from shu_insight.processing.dates import SHANGHAI_TZ
from shu_insight.ranking.explain import explain
from shu_insight.ranking.score import ScoreWeights

SOURCE_NAMES = {
    "shu_main": "上海大学官网",
    "undergraduate": "上海大学本科生院",
    "graduate": "上海大学研究生院",
    "computer": "计算机工程与科学学院",
}

ROBOTS_DIAGNOSTIC_LABELS = {
    "robots_unavailable": "robots 文件不可用",
    "robots_not_found": "robots.txt 返回 404/410（未找到）",
    "robots_forbidden": "robots.txt 返回 401/403（访问受限）",
    "robots_rate_limited": "robots.txt 返回 429（访问过频）",
    "robots_other_4xx": "robots.txt 返回其他 4xx",
    "robots_server_error": "robots.txt 返回 5xx",
    "robots_redirect": "robots.txt 返回跳转，目标未核验",
    "robots_unexpected_status": "robots.txt 返回非预期状态",
    "robots_wrong_content_type": "robots.txt 非纯文本类型",
    "robots_html_or_login": "robots.txt 返回网页或登录页",
    "robots_missing_directives": "robots.txt 缺少可解析的访问规则",
}

TRANSPORT_STAGE_LABELS = {
    "robots": "robots.txt", "list": "栏目页", "detail": "正文页",
}
TRANSPORT_KIND_LABELS = {
    "proxy": "代理连接失败", "connect": "连接失败",
    "connect_timeout": "连接超时", "read": "读取失败",
    "read_timeout": "读取超时", "other": "传输失败",
    "remote_protocol": "远端协议错误", "local_protocol": "本地协议错误",
    "write": "写入失败", "write_timeout": "写入超时", "pool_timeout": "连接池等待超时",
    "close": "关闭连接失败", "unsupported_protocol": "不支持的协议",
}
TRANSPORT_DIAGNOSTIC_LABELS = {
    f"transport_{stage}_{kind}": f"{stage_label}{' ' if stage == 'robots' else ''}{kind_label}"
    for stage, stage_label in TRANSPORT_STAGE_LABELS.items()
    for kind, kind_label in TRANSPORT_KIND_LABELS.items()
}

SOURCE_STATE_LABELS = {
    "success": "正常",
    "not_modified": "正常（无更新）",
    "rate_limited": "受限",
    "structure_changed": "列表结构变化",
    "forbidden": "禁止访问",
    "robots_disallowed": "robots 禁止",
    **ROBOTS_DIAGNOSTIC_LABELS,
    "connection_failed": "连接学校服务器失败",
    "connection_timeout": "连接学校服务器超时",
    "circuit_open": "访问已暂停",
    "redirect_not_allowed": "跳转超出允许范围",
    "authentication_required": "需要身份认证",
    "login_or_captcha": "登录或验证码拦截",
    "not_modified_without_body": "页面未返回可核对内容",
    "http_error": "网页请求失败",
    "unsupported_content_type": "网页类型不支持",
    "unresolved_redirect": "跳转未完成",
    "invalid_url": "网址无效",
    "https_required": "必须使用 HTTPS",
    "host_not_allowed": "来源域名不符",
    "path_not_allowed": "路径不在允许范围",
    "attachment_blocked": "附件不采集",
    "login_required": "需要登录",
    "error": "异常",
    "unknown": "尚未检查",
    "not_checked": "尚未检查官网",
    "synthetic_demo": "尚未检查官网",
    "partial": "部分检查",
    "config_review_required": "配置需重新审阅",
    "interrupted": "检查中断",
    "request_budget_exhausted": "请求预算已用尽",
}

UPDATE_REASON_LABELS = {
    **SOURCE_STATE_LABELS,
    **TRANSPORT_DIAGNOSTIC_LABELS,
    "robots_not_published": "robots.txt 未发布（404/410），按协议继续检查公开栏目",
    "pagination_unverified": "分页未验证（仅第一页）",
    "page_limit_reached": "已达本轮列表页上限",
    "reviewed_range_complete": "已处理本轮审阅范围",
    "pagination_failed": "分页结构未通过核验",
    "pagination_loop": "分页出现循环",
    "pagination_not_allowed": "分页链接超出允许范围",
    "list_check_incomplete": "本轮栏目检查未完成",
    "pending_backlog": "尚有未核实候选积压",
    "queue_full": "候选队列已满",
    "queue_expired": "候选已超过 90 天",
    "queue_paused": "候选暂缓重试",
    "queue_removed": "失效或不允许的候选已移除",
    "sensitive_content": "正文包含不采集内容",
    "sensitive_title": "标题包含不采集内容",
    "content_unchanged": "正文无变化",
    "unchanged_list": "栏目页无变化",
    "unchanged_detail": "正文页无变化",
    "not_found": "正文不存在",
    "candidate_limit_reached": "已达本轮候选处理上限",
    "temporary_failure": "临时访问失败",
    "other": "其他安全限制",
}

SKIP_REASON_LABELS = {
    **UPDATE_REASON_LABELS,
    # The pipeline records this reason for a detail page it could not parse.
    # A listing-page failure is reported separately as the source state.
    "structure_changed": "正文页结构无法解析",
}


def _source_status_reason_label(state: str, reason: str) -> str:
    labels = SKIP_REASON_LABELS if state == "partial" else UPDATE_REASON_LABELS
    return labels.get(reason, "")


def manual_update_progress_text(event: UpdateEvent, source_names: Mapping[str, str]) -> str | None:
    labels = {
        "source_started": "开始检查", "list_page_read": "已读取栏目页",
        "notice_saved": "已保存核实通知", "notice_skipped": "已跳过候选",
        "source_failed": "本来源检查停止", "source_finished": "本轮范围检查完成",
    }
    if event.source_id not in source_names or event.kind not in labels:
        return None
    text = f"{source_names[event.source_id]}：{labels[event.kind]}"
    reason_labels = SKIP_REASON_LABELS if event.kind == "notice_skipped" else UPDATE_REASON_LABELS
    if event.kind in {"source_failed", "notice_skipped"} and event.message in reason_labels:
        text += " · " + reason_labels[event.message]
    return text


def _update_time(value: Any) -> str:
    try:
        timestamp = datetime.fromisoformat(value) if isinstance(value, str) else value
        if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
            return "尚无记录"
        return timestamp.astimezone(SHANGHAI_TZ).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return "尚无记录"


def render_manual_run_status(latest: Mapping[str, Any] | None) -> None:
    if latest is None:
        st.caption("最近一次手动点击：尚无记录")
        return
    label = {"success": "本轮范围检查完成", "partial": "部分检查",
             "failed": "未完成检查", "interrupted": "检查中断"}.get(
                 latest.get("state"), "未完成检查")
    if latest.get("finished_at") is None:
        label = "尚无结束记录（可能其他页面仍在更新，或上次运行中断）"
    st.caption(f"最近一次手动点击：{_update_time(latest.get('started_at'))} · {label}")
    if latest.get("finished_at") is not None:
        st.caption(f"结束时间：{_update_time(latest.get('finished_at'))}")


def render_manual_update_result(report: UpdateReport) -> None:
    if report.state == "concurrent_skipped":
        st.warning("已有更新正在运行；本次点击已跳过，请稍后刷新状态。")
        return
    if not report.by_source:
        st.error("没有可更新的官网或本科生院来源，请核对本地来源配置。"
                 if report.message == "no_eligible_sources" else "本次未完成检查，请稍后重试。")
        return
    source_count = len(report.by_source)
    updated = sum(result.updated_count for result in report.by_source.values())
    counts = (f"本次写入 {report.saved_count} 条 · 新增 {report.saved_count - updated} 条"
              f" · 更新 {updated} 条；请核对逐站范围和安全限制。")
    if report.state == "success":
        st.success(f"{source_count} 个来源均完成检查；" + counts)
    elif report.state == "partial":
        completed = sum(result.state == "success" for result in report.by_source.values())
        prefix = ("部分来源未完成" if completed else f"{source_count} 个来源均未完成检查")
        st.warning("部分检查；" + prefix + "；" + counts)
    else:
        st.error(f"{source_count} 个来源均未完成检查；" + counts)
    for source_id, result in report.by_source.items():
        if source_id not in SOURCE_NAMES:
            continue
        label = {"success": "已检查"}.get(
            result.state, SOURCE_STATE_LABELS.get(result.state, "未完成检查"))
        st.caption(f"{SOURCE_NAMES[source_id]}：{label} · "
                   f"新增 {result.saved_count - result.updated_count} 条 · "
                   f"更新 {result.updated_count} 条 · 跳过 {result.skipped_count} 条")
        st.caption(f"正文已读取 {result.parsed_count} 条 · 仅列表可见 {result.listing_count} 条 · "
                   f"待处理 {result.pending_count} 条（其中仅 URL {result.queued_count} 条）")
        reason_label = _source_status_reason_label(result.state, result.message)
        if reason_label:
            st.caption("安全限制或结果：" + reason_label)
        with st.expander(f"{SOURCE_NAMES[source_id]}：本轮技术计数"):
            _render_request_counters(result.request_breakdown)
            st.caption(f"发现 {result.discovered_count} 条 · 读页 {result.list_pages_read} · "
                       f"物理请求 {result.request_count}")
        diagnostic = result.listing_diagnostic
        if diagnostic is not None:
            st.caption(
                "栏目响应诊断（仅本会话可见，不保存网页）："
                f"HTML 字符 {diagnostic.html_characters} 个 · "
                f"网页链接 {diagnostic.link_count} 个 · "
                f"同站 /info/ 链接 {diagnostic.scoped_article_link_count} 个 · "
                f"当前规则匹配 {diagnostic.selected_anchor_count} 个"
            )
            for pattern in diagnostic.link_patterns:
                raw_label = "含 /info/" if pattern.raw_info_marker else "不含 /info/"
                date_label = "邻近日期" if pattern.nearby_date else "未见邻近日期"
                st.caption(
                    f"链接结构 ×{pattern.count}：{pattern.ancestor_signature}"
                    f" · 原链接{raw_label} · {date_label}"
                )
            hidden_count = diagnostic.scoped_article_link_count - sum(
                pattern.count for pattern in diagnostic.link_patterns
            )
            if hidden_count > 0:
                st.caption(f"另有 {hidden_count} 个同站 /info/ 链接的结构未展开。")


def _render_request_counters(breakdown: Mapping[str, int]) -> None:
    st.caption(f"robots {breakdown.get('robots', 0)} · 列表 {breakdown.get('list', 0)} · "
               f"正文 {breakdown.get('detail', 0)} · 重试 {breakdown.get('retry', 0)}")


def render_manual_source_status(
    source_name: str,
    *,
    status: Mapping[str, Any] | None = None,
    last_list_check: Mapping[str, Any] | None = None,
    last_complete_check: Mapping[str, Any] | None = None,
    queued_count: int = 0,
    pending_count: int = 0,
) -> None:
    status = status or {}
    state = status.get("state", "not_checked")
    label = {"success": "已检查", "unknown": "未检查", "not_checked": "未检查",
             "synthetic_demo": "未检查"}.get(state, SOURCE_STATE_LABELS.get(state, "未完成检查"))
    saved = status.get("saved_count", 0)
    updated = status.get("updated_count", 0)
    breakdown = status.get("request_breakdown") or {}
    legacy_counters = status.get("request_count", 0) > 0 and not any(
        breakdown.get(stage, 0) for stage in ("robots", "list", "detail")
    )
    with st.container(border=True):
        st.caption(f"{source_name}：{label} · 新增 {saved - updated} 条 · 更新 {updated} 条")
        st.caption(f"最近尝试：{_update_time(status.get('checked_at'))}")
        st.caption("最近成功读取栏目页：" + _update_time(
            (last_list_check or {}).get("checked_at")))
        complete = last_complete_check or {}
        scope = ""
        complete_time = _update_time(complete.get("checked_at"))
        if complete:
            if not complete.get("run_id"):
                st.caption("历史成功检查：" + complete_time + "（旧记录未记录范围）")
                complete_time = "尚无记录"
            elif complete.get("coverage_code") == "pagination_unverified":
                scope = "（仅第一页，分页未验证）"
            elif complete.get("coverage_code") == "page_limit_reached":
                scope = "（已达当轮列表页上限）"
            elif complete.get("coverage_code") != "reviewed_range_complete":
                st.caption("历史成功检查：" + complete_time + "（旧记录未记录范围）")
                complete_time = "尚无记录"
        st.caption("最近本轮范围完整处理：" + complete_time + scope)
        if legacy_counters:
            st.caption("历史更新记录：新版本详细计数将在下次手动更新后显示")
        else:
            st.caption(f"正文已读取 {status.get('parsed_count', 0)} 条 · "
                       f"仅列表可见 {status.get('listing_count', 0)} 条")
        st.caption(f"最近尝试结束时积压：{status.get('queued_count', 0)} 条（仅 URL）")
        st.caption(f"当前未核实队列：{queued_count} 条（不进入通知中心）")
        st.caption(f"当前待正文处理：{pending_count} 条（可包含列表卡片）")
        with st.expander("技术计数"):
            st.caption(f"最近尝试读页数：{status.get('list_pages_read', 0)} · "
                       f"物理请求数：{status.get('request_count', 0)} · "
                       f"跳过：{status.get('skipped_count', 0)} 条")
            if not legacy_counters:
                st.caption(f"本轮发现：{status.get('discovered_count', 0)} 条")
                _render_request_counters(breakdown)
        coverage = status.get("coverage_code")
        if not status.get("list_pages_read") and coverage in {
            "pagination_unverified", "pagination_failed", "page_limit_reached",
            "reviewed_range_complete", "list_check_incomplete",
        }:
            coverage = ""  # Earlier versions could mislabel a pre-list failure.
        if coverage in UPDATE_REASON_LABELS:
            st.caption("检查范围：" + UPDATE_REASON_LABELS[coverage])
        reason = status.get("message")
        reason_label = _source_status_reason_label(state, reason)
        if reason_label:
            st.caption("安全限制或结果：" + reason_label)
        if reason in ROBOTS_DIAGNOSTIC_LABELS or state in ROBOTS_DIAGNOSTIC_LABELS:
            st.caption("浏览器能打开网页，不代表程序能核验 robots.txt；本来源已停止抓取。")
        elif reason in TRANSPORT_DIAGNOSTIC_LABELS or reason in {
            "connection_failed", "connection_timeout",
        } or state in {
            "connection_failed", "connection_timeout",
        }:
            st.caption("浏览器与程序可能使用不同的代理或证书设置；请检查本机连接。")
        reasons = status.get("skip_reasons") or {}
        if reasons:
            st.caption("跳过原因：" + "、".join(
                f"{SKIP_REASON_LABELS.get(key, '其他安全限制')} {count}"
                for key, count in sorted(reasons.items())
            ))


def stale_source_ids(registry, today: date) -> list[str]:
    return [source_id for source_id, adapter in registry.items()
            if (today - adapter.config.reviewed_on).days >= 90]


def _date(
    value: datetime | None,
    *,
    deadline_confidence: DeadlineConfidence | None = None,
) -> str:
    if value is None:
        return "未知"
    local = value.astimezone(SHANGHAI_TZ)
    date_only = (
        deadline_confidence in {DeadlineConfidence.LOW, DeadlineConfidence.UNKNOWN}
        if deadline_confidence is not None else local.hour == local.minute == 0
    )
    return local.strftime("%Y-%m-%d" if date_only else "%Y-%m-%d %H:%M")


def _bar(value: float, width: int = 12) -> str:
    filled = round(min(1.0, max(0.0, value)) * width)
    return "█" * filled + "░" * (width - filled)


def _render_score_breakdown(recommendation: Recommendation, weights: ScoreWeights) -> None:
    """Show how the five weighted parts add up to the displayed total."""
    score = recommendation.score
    rows = [
        ("相关度", score.relevance, weights.relevance),
        ("紧迫程度", score.urgency, weights.urgency),
        ("来源权威", score.authority, weights.authority),
        ("新鲜度", score.freshness, weights.freshness),
    ]
    with st.expander("为什么排在这里？"):
        for label, value, weight in rows:
            if weight <= 0:
                st.markdown(f"`{label}` {_bar(0.0)} 本模式不计分（权重 0）")
                continue
            st.markdown(
                f"`{label}` {_bar(value)} **{value:.2f}** × {weight:.2f} "
                f"= `{value * weight:.3f}`"
            )
        if weights.audience <= 0:
            st.markdown(f"`受众契合` {_bar(0.0)} 本模式不计分（权重 0）")
        elif not score.audience_known:
            st.markdown(f"`受众契合` {_bar(0.5)} 通知未写明适用对象，本维度无证据")
            st.caption("该维度不参与本次总分，其余四项权重按比例归一化。")
        else:
            st.markdown(
                f"`受众契合` {_bar(score.audience)} **{score.audience:.2f}** "
                f"× {weights.audience:.2f} = `{score.audience * weights.audience:.3f}`"
            )
            structural = (
                f"{score.audience_structural:.2f}（学历/年级/学院/专业）"
                if score.audience_structural_known else "无明确受众信息"
            )
            theme = (
                f"{score.audience_theme:.2f}（画像主题与通知标签的重合度）"
                if score.audience_theme_known else "无画像主题"
            )
            st.caption(
                f"结构契合 {structural} × 0.7 + 主题契合 {theme} × 0.3；"
                "某一侧缺失时由另一侧归一。"
            )
        st.caption(f"总分 **{score.total:.3f}**：先按层级分组（直接相关 > 可能适用 > 其他通知），"
                   "同层级内再比总分；层级由标题/公开摘要的命中证据决定，不随画像改变。")
        if recommendation.match_tier is MatchTier.POSSIBLE:
            st.caption("“可能适用”表示只有摘要命中或仅列表可见，正文证据不足，请以官网原文为准。")


def render_notice_card(
    notice: NoticeRecord,
    *,
    recommendation: Recommendation | None = None,
    sort_basis: str = "",
    ask_notice: Callable[[NoticeRecord], None] | None = None,
    weights: ScoreWeights | None = None,
    on_read: Callable[[NoticeRecord], object] | None = None,
) -> None:
    listing_only = notice.parse_state is ParseState.LISTING_ONLY
    with st.container(border=True):
        if (on_read is not None and notice.canonical_url.startswith("https://")
                and "synthetic_demo" not in notice.topic_tags):
            st.link_button(f"**{notice.title}**", notice.canonical_url,
                           key=f"notice-title-{notice.id}", type="tertiary", width="stretch",
                           wrap=True,
                           on_click=on_read, args=(notice,))
        elif notice.canonical_url.startswith("https://"):
            st.markdown(f"### [{notice.title}]({notice.canonical_url})")
        else:
            st.markdown(f"### {notice.title}")
        source = SOURCE_NAMES.get(notice.source_id, notice.source_id)
        deadline_label = _date(
            notice.deadline_at, deadline_confidence=notice.deadline_confidence
        )
        if notice.deadline_at and notice.deadline_confidence in {
            DeadlineConfidence.LOW,
            DeadlineConfidence.UNKNOWN,
        }:
            deadline_label += "（待核实）"
        published_label = ("发布日期未知" if listing_only and notice.published_at is None
                           else f"发布：{_date(notice.published_at)}")
        st.caption(f"来源：{source} · {published_label} · 截止：{deadline_label}")
        if notice.verified_at is not None:
            if listing_only:
                st.caption(f"仅官网列表可见 · 列表核对于 {_date(notice.verified_at)}")
            else:
                proof_label = (
                    "官网快照" if notice.proof_status is ProofStatus.VERIFIED_SNAPSHOT
                    else "官网更新"
                )
                st.caption(f"{proof_label} · 核对于 {_date(notice.verified_at)}")
        today = datetime.now(SHANGHAI_TZ).date()
        expired = bool(notice.deadline_at and (
            notice.deadline_at.astimezone(SHANGHAI_TZ).date() < today
            if notice.deadline_confidence is DeadlineConfidence.LOW else
            notice.deadline_at < datetime.now(SHANGHAI_TZ)
        ))
        old = bool(notice.published_at and
                   notice.published_at.astimezone(SHANGHAI_TZ).date() < today - timedelta(days=90))
        if expired or old:
            st.caption("历史通知 · 时限或发布已久，请核对官网后续信息")
        if notice.categories and not listing_only:
            st.markdown(" · ".join(f"`{category}`" for category in notice.categories))
        if listing_only:
            st.write("正文尚未读取，暂无摘要")
        elif not notice.summary:
            st.caption("正文已读取，摘要未生成")
            st.write("具体内容请查看官网原文")
        else:
            st.write(notice.summary)
        if recommendation is not None:
            st.caption({
                MatchTier.DIRECT: "直接相关",
                MatchTier.POSSIBLE: "可能适用",
                MatchTier.OTHER: "其他通知",
            }[recommendation.match_tier])
            if listing_only and any(
                reason.startswith("匹配关键词：") for reason in recommendation.reasons
            ):
                st.caption("仅标题匹配，正文待读取")
            st.info(explain(recommendation))
            if weights is not None:
                _render_score_breakdown(recommendation, weights)
        elif sort_basis:
            st.caption(f"排序依据：{sort_basis}")
        if "synthetic_demo" in notice.topic_tags:
            st.caption("虚构演示通知，无官网原文；来源名称仅为演示角色。")
        else:
            original, question = st.columns([3, 2], vertical_alignment="bottom")
            with original:
                st.link_button("查看官网原文", notice.canonical_url,
                               key=f"notice-original-{notice.id}",
                               on_click=on_read if on_read is not None else "ignore",
                               args=(notice,))
            if ask_notice is not None:
                with question:
                    if st.button("问问这条通知", key=f"notice-qa-{notice.id}", width="stretch"):
                        ask_notice(notice)


def render_source_health_card(
    source_name: str,
    *,
    state: str = "unknown",
    checked_at: str | None = None,
    message: str = "",
    saved_count: int = 0,
    updated_count: int = 0,
    skipped_count: int = 0,
    skip_reasons: dict[str, int] | None = None,
    visible_count: int = 0,
) -> None:
    label = SOURCE_STATE_LABELS.get(state, "异常")
    with st.container(border=True):
        st.markdown(f"**{source_name}**")
        st.metric("最近状态", label)
        st.caption(f"检查时间：{checked_at or '尚无记录'}")
        if state == "success" and visible_count == 0:
            st.caption("访问成功，0 条已核实")
        elif visible_count:
            st.caption(f"当前已核实通知：{visible_count} 条")
        st.caption(f"本次写入：{saved_count} · 更新：{updated_count} · 跳过：{skipped_count}")
        if skip_reasons:
            safe_reasons = [
                f"{reason if reason in SKIP_REASON_LABELS else '其他安全限制'} {count}"
                for reason, count in sorted(skip_reasons.items())
                if isinstance(count, int) and not isinstance(count, bool) and count > 0
            ]
            if safe_reasons:
                st.caption("跳过原因：" + "、".join(safe_reasons))
        reason_label = _source_status_reason_label(state, message)
        if reason_label:
            st.caption("说明：" + reason_label)
