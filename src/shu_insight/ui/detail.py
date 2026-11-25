import json

import streamlit as st

from shu_insight.domain.models import NoticeRecord, ParseState


def render_detail(notice: NoticeRecord, evidence: dict | None = None) -> None:
    if notice.parse_state is ParseState.LISTING_ONLY:
        return
    with st.expander("结构化详情"):
        st.write(
            {
                "适用层次": notice.education_levels,
                "适用年级": notice.grades,
                "学院": notice.colleges,
                "专业": notice.majors,
                "主题标签": notice.topic_tags,
                "地点": notice.location or "未知",
            }
        )
        if evidence:
            for key, label in (("duplicates", "重复来源"),
                               ("possible_duplicates", "可能重复（未合并）")):
                st.markdown(f"**{label}**")
                for item in evidence[key]:
                    other = item["notice"]
                    st.markdown(f"[{other.title}]({other.canonical_url}) · {other.source_id}")
                    st.caption(f"相似度：{item['evidence']['score']:.3f}")
            st.markdown("**版本变化**")
            for version in evidence["versions"]:
                st.caption(version["created_at"])
                changes = version.get("changes", [])
                if changes:
                    display_changes = [
                        {key: value if isinstance(value, str) else json.dumps(
                            value, ensure_ascii=False, sort_keys=True,
                        ) for key, value in change.items()}
                        for change in changes
                    ]
                    st.dataframe(display_changes, hide_index=True)
                else:
                    st.caption("初始记录或仅内容指纹变化")
