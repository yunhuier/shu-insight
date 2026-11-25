"""A per-notice dialog; no model request occurs until explicit form submission."""
from __future__ import annotations

import hashlib
import os

import streamlit as st

from shu_insight.domain.models import NoticeRecord
from shu_insight.services.notice_fulltext import prepare_fulltext
from shu_insight.services.notice_qa import DeepSeekClient, NoticeQaError, retrieve_evidence
from shu_insight.services.notices import NoticeService


@st.dialog("通知信息问答", width="small")
def render_notice_qa(notice: NoticeRecord, service: NoticeService) -> None:
    st.markdown(f"**{notice.title}**")
    st.caption("提问时读取官网正文，检索相关全文片段回答；未读到全文会明确提示。")
    with st.expander("DeepSeek 设置", expanded=not (
        st.session_state.get("deepseek-api-key") or os.getenv("DEEPSEEK_API_KEY"))):
        st.text_input("DeepSeek 官方 API Key", type="password", key="deepseek-api-key",
                      persist_state="session",
                      help="仅保存在本次浏览器会话中，不写入项目文件或数据库。")
        if os.getenv("DEEPSEEK_API_KEY"):
            st.caption("启动环境已配置 Key；此处留空会使用环境中的 Key。")
        st.caption("发送问题时，问题、最近对话和检索到的公开片段会发送到 DeepSeek。")
    api_key = (st.session_state.get("deepseek-api-key", "").strip()
               or os.getenv("DEEPSEEK_API_KEY", "").strip())
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip() or "deepseek-flash"
    related = st.checkbox("同时参考相关通知", value=True, key=f"qa-related-{notice.id}")
    use_fulltext = st.checkbox("读取官网全文", value=True, key=f"qa-fulltext-{notice.id}",
                              help="发送问题时读取当前及最多两条已收录相关通知的官网正文。")
    fulltext_cache = st.session_state.setdefault("notice-fulltext-cache", {})
    if st.button("清除全文缓存", key=f"qa-refresh-{notice.id}"):
        fulltext_cache.clear()
        st.caption("已清除，下次发送问题时重新读取官网。")
    signature = hashlib.sha256((notice.title + notice.summary + notice.content_fingerprint
                                + str(notice.deadline_at)).encode()).hexdigest()
    conversations = st.session_state.setdefault("notice-qa-conversations", {})
    if conversations.get(notice.id, {}).get("signature") != signature:
        if len(conversations) >= 32:
            conversations.pop(next(iter(conversations)))
        conversations[notice.id] = {"signature": signature, "turns": []}
    conversation = conversations[notice.id]
    if st.button("清空本条对话", key=f"qa-clear-{notice.id}"):
        conversation["turns"] = []
    chat = st.container()
    if not api_key:
        st.info("展开 DeepSeek 设置填写 Key 后即可提问。无需把 Key 发到聊天中。")
    with st.form(f"qa-form-{notice.id}"):
        question = st.text_area("问题", max_chars=1000, key=f"qa-question-{notice.id}",
                                placeholder="例如：这条通知主要讲什么？申请时间是什么时候？")
        submitted = st.form_submit_button("发送问题", disabled=not api_key)
    if submitted:
        if not question.strip():
            st.info("请先输入问题。")
        else:
            turns = conversation["turns"]
            history = [message for turn in turns[-3:] for message in (
                {"role": "user", "content": turn["question"]},
                {"role": "assistant", "content": turn["answer"]},
            )]
            retrieval_query = " ".join([question] + [turn["question"] for turn in turns[-2:]])
            evidence = retrieve_evidence(service.all(), notice.id, retrieval_query, related=related)
            try:
                with st.spinner("正在读取、检索并生成回答…"):
                    if use_fulltext:
                        progress = st.empty()
                        fetched = prepare_fulltext(evidence, retrieval_query, fulltext_cache,
                                                   on_progress=progress.caption)
                        evidence = fetched.evidence
                        progress.caption(f"已取得 {fetched.fulltext_count} 篇官网正文"
                                         f"（其中缓存 {fetched.cached_count} 篇）")
                        for warning in fetched.warnings:
                            st.warning(warning)
                    answer = DeepSeekClient(api_key, model=model).answer(
                        question, evidence, history)
                turns.append({"question": question, "answer": answer.answer,
                              "sources": answer.sources,
                              "warnings": fetched.warnings if use_fulltext else ()})
                conversation["turns"] = turns[-6:]
            except NoticeQaError as error:
                st.error(str(error))
    with chat:
        for turn in conversation["turns"]:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                for warning in turn.get("warnings", ()):
                    st.caption(warning)
                st.write(turn["answer"])
                with st.expander("查看回答依据"):
                    for item in turn["sources"]:
                        label = "当前通知" if item.notice_id == notice.id else "补充通知"
                        st.markdown(f"{label}：[{item.title}]({item.url})")
                        st.caption(item.kind)
                        st.text(item.text)
