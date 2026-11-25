from __future__ import annotations

import streamlit as st

from shu_insight.domain.models import NoticeRecord, ParseState
from shu_insight.profiles.learning import ArticleReadLearner
from shu_insight.services.notices import NoticeQuery, NoticeService
from shu_insight.ui.components import render_notice_card
from shu_insight.ui.detail import render_detail
from shu_insight.ui.notice_qa import render_notice_qa
from shu_insight.ui.personas import PersonaSelection


def _empty(message: str) -> None:
    st.info(message)


def _remember_browse_mode() -> None:
    st.session_state["selected-browse-mode"] = st.session_state["browse-mode"]


def _remember_timeline_order() -> None:
    st.session_state["selected-timeline-order"] = st.session_state["timeline-order"]


def _matches_search(notice: NoticeRecord, terms: tuple[str, ...]) -> bool:
    text = notice.title if notice.parse_state is ParseState.LISTING_ONLY else (
        f"{notice.title}\n{notice.summary}")
    return not terms or any(term.casefold() in text.casefold() for term in terms)


def _result_count(count: int, terms: tuple[str, ...]) -> None:
    if terms:
        st.caption("关键词搜索：" + "、".join(terms) + f" · 命中 {count} 条通知")
    else:
        st.caption(f"当前显示 {count} 条通知")


def render_home(
    service: NoticeService,
    selection: PersonaSelection,
    *,
    restore_widgets: bool = False,
    learner: ArticleReadLearner | None = None,
) -> None:
    st.html("""
        <style>
        [class*="st-key-notice-title-"] a {
            justify-content: flex-start;
            text-align: left;
            height: auto;
            padding: 0.25rem 0;
            color: #0068c9;
            text-decoration: underline;
        }
        [class*="st-key-notice-title-"] a,
        [class*="st-key-notice-title-"] a [data-testid="stMarkdownContainer"],
        [class*="st-key-notice-title-"] a p,
        [class*="st-key-notice-title-"] a strong {
            font-size: 1.75rem !important;
            font-weight: 600 !important;
            line-height: 1.2 !important;
            letter-spacing: -0.005em;
        }
        </style>
    """)

    def ask_notice(notice: NoticeRecord) -> None:
        render_notice_qa(notice, service)

    on_read = learner.record_read if learner is not None else None
    modes = ("全部通知", "为你推荐")
    st.session_state.setdefault("selected-browse-mode", "为你推荐")
    if restore_widgets or "browse-mode" not in st.session_state:
        st.session_state["browse-mode"] = st.session_state["selected-browse-mode"]
    mode = st.radio(
        "浏览方式",
        modes,
        horizontal=True,
        key="browse-mode",
        on_change=_remember_browse_mode,
        persist_state="session",
    )
    if mode == "为你推荐":
        card_weights = (
            service.ranking_weights.profile if selection.persona is not None
            else service.ranking_weights.no_profile
        )
        if selection.persona is not None:
            st.caption(
                f"画像模式：按「{selection.persona_label}」排序，受众契合占 25%"
                "（通用排序为 0；其中结构 17.5%、主题 7.5%；通知未写明适用对象时"
                "该维度不参与，其余权重按比例归一）。"
                "先按相关层级（直接相关/可能适用）分组，再比总分。"
            )
            if not selection.terms:
                st.caption(
                    "尚未选择兴趣标签或关键词：当前排序主要反映“通知是否明确面向你”，"
                    "建议先选标签，再叠加画像。"
                )
        recommendations = service.recommend(
            NoticeQuery(terms=selection.terms, persona=selection.persona)
        )
        recommendations = [item for item in recommendations
                           if _matches_search(item.notice, selection.search_terms)]
        _result_count(len(recommendations), selection.search_terms)
        if not recommendations:
            _empty("没有匹配关键词的通知；可清空关键词查看全部通知。"
                   if selection.search_terms else "尚无已核实的官网通知。")
        for recommendation in recommendations:
            render_notice_card(recommendation.notice, recommendation=recommendation,
                               ask_notice=ask_notice, weights=card_weights, on_read=on_read)
            render_detail(recommendation.notice, service.detail_evidence(recommendation.notice.id))
    else:
        st.session_state.setdefault("selected-timeline-order", "最新发布")
        if restore_widgets or "timeline-order" not in st.session_state:
            st.session_state["timeline-order"] = st.session_state["selected-timeline-order"]
        order = st.radio(
            "通知排序",
            ("最新发布", "即将截止"),
            horizontal=True,
            key="timeline-order",
            on_change=_remember_timeline_order,
            persist_state="session",
        )
        timeline = service.upcoming() if order == "即将截止" else None
        notices = service.latest() if timeline is None else timeline.dated
        notices = [notice for notice in notices if _matches_search(notice, selection.search_terms)]
        unknown = [notice for notice in timeline.unknown
                   if _matches_search(notice, selection.search_terms)] if timeline else []
        expired = [notice for notice in timeline.expired
                   if _matches_search(notice, selection.search_terms)] if timeline else []
        _result_count(len(notices) + len(unknown) + len(expired), selection.search_terms)
        if timeline is not None:
            st.subheader("已知截止时间")
        if not notices and selection.search_terms and not unknown and not expired:
            _empty("没有匹配关键词的通知；可清空关键词查看全部通知。")
        elif not notices:
            if timeline is not None and (unknown or expired):
                _empty("当前没有尚未截止且日期已知的官网通知。")
            else:
                _empty("尚无已核实的官网通知。")
        for notice in notices:
            render_notice_card(notice, sort_basis=order, ask_notice=ask_notice, on_read=on_read)
            render_detail(notice, service.detail_evidence(notice.id))
        if timeline is not None:
            st.subheader("截止时间未知")
            for notice in unknown:
                render_notice_card(notice, sort_basis="截止时间未知，请查阅原文",
                                   ask_notice=ask_notice, on_read=on_read)
                render_detail(notice, service.detail_evidence(notice.id))
            st.subheader("已过截止")
            for notice in expired:
                render_notice_card(notice, sort_basis="已过截止", ask_notice=ask_notice,
                                   on_read=on_read)
                render_detail(notice, service.detail_evidence(notice.id))
