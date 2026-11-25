from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import streamlit as st

from shu_insight.domain.models import Persona
from shu_insight.processing.classify import Taxonomy
from shu_insight.profiles.learning import (
    LEARN_LABEL,
    LEARN_PERSONA_ID,
    LEARN_UNAVAILABLE_SUFFIX,
    ArticleReadLearner,
)
from shu_insight.ranking.expand import Query, expand_terms

TAGS_KEY = "interest-tags"
KEYWORDS_KEY = "search-keywords"
PERSONA_KEY = "persona-index"

NO_PERSONA_LABEL = "不使用画像（通用排序）"
CLUSTER_INDEX = 1


@dataclass(frozen=True, slots=True)
class PersonaSelection:
    persona: Persona | None
    terms: tuple[str, ...]
    search_terms: tuple[str, ...] = ()
    persona_label: str = ""
    tags: tuple[str, ...] = ()


def clear_keywords() -> None:
    st.session_state[KEYWORDS_KEY] = ""


def _persona_caption(persona: Persona) -> str:
    parts = [
        part
        for part in (persona.education_level, persona.grade, persona.college, persona.major)
        if part and part not in {"不限", "不限专业"}
    ]
    scope = "·".join(parts) if parts else "受众不限"
    goals = "、".join(persona.goals) if persona.goals else "未设置目标"
    caption = f"{scope} · 目标：{goals}"
    if persona.interest_tags:
        caption += f" · 主题：{'、'.join(persona.interest_tags[:4])}"
    return caption


def render_interest_panel(
    taxonomy: Taxonomy,
    personas: Sequence[Persona] = (),
    *,
    learner: ArticleReadLearner | None = None,
) -> PersonaSelection:
    st.sidebar.header("兴趣标签")
    tags = tuple(st.sidebar.multiselect(
        "兴趣标签（最多 8 个）",
        tuple(dict.fromkeys(("保研", *taxonomy.tags, *taxonomy.categories))),
        default=[], max_selections=8, key=TAGS_KEY, persist_state="session",
        help="选择感兴趣的主题；为你推荐会将相关通知排在前面。",
    ))
    st.sidebar.header("关键词搜索")
    raw = st.sidebar.text_input(
        "关键词（最多 5 个）", key=KEYWORDS_KEY,
        help="用中文或英文逗号分隔；两种浏览方式均显示标题或公开摘要命中任一关键词的通知。",
        persist_state="session",
    )
    unique: dict[str, str] = {}
    for item in raw.replace(",", "，").split("，"):
        if term := item.strip():
            unique.setdefault(term.casefold(), term)
    keywords = tuple(unique.values())[:5]
    if len(unique) > 5:
        st.sidebar.caption("仅使用前 5 个关键词。")
    if raw:
        st.sidebar.button("清空关键词", on_click=clear_keywords)
    st.sidebar.header("我的画像")
    read_count = learner.read_count() if learner is not None else 0
    cluster_ready = learner is not None and read_count >= learner.minimum_articles
    options = [NO_PERSONA_LABEL]
    if learner is not None:
        options.append(
            LEARN_LABEL if cluster_ready else f"{LEARN_LABEL}{LEARN_UNAVAILABLE_SUFFIX}"
        )
    first_persona_index = len(options)
    options.extend(persona.label for persona in personas)
    if len(set(options)) != len(options):
        st.sidebar.caption("存在重名画像，按列表顺序区分。")
    if (
        learner is not None
        and not cluster_ready
        and st.session_state.get(PERSONA_KEY) == CLUSTER_INDEX
    ):
        # The cluster option stays listed but cannot be selected before it has
        # enough history: bounce the selection back to the general ranking.
        st.session_state[PERSONA_KEY] = 0
    index = st.sidebar.selectbox(
        "按画像排序（可选）",
        range(len(options)),
        format_func=lambda value: options[value],
        key=PERSONA_KEY,
        persist_state="session",
        help=(
            "画像只影响“为你推荐”的排序：受众契合由两部分合成——结构契合看通知是否"
            "明确面向你的培养层次、年级、学院、专业；主题契合看通知标签与你画像主题的"
            "重合度。通知没写适用对象时该维度不参与，其余权重按比例归一。"
            "通用排序不计算这一项。聚类画像依据实际点开的不同文章内容生成，至少需要 10 篇。"
        ),
    )
    persona: Persona | None = None
    if learner is not None and not cluster_ready:
        st.sidebar.caption(
            f"聚类画像暂不可用：请点击文章标题或“查看官网原文”，"
            f"至少打开 {learner.minimum_articles} 篇不同文章。"
            f"已打开 {read_count} 篇，还需 {learner.minimum_articles - read_count} 篇。"
        )
    if learner is not None:
        st.sidebar.caption("按不同文章计数，同篇重复点击只计一篇；记录保存在本机。")
    if learner is not None and index == CLUSTER_INDEX and cluster_ready:
        learned = learner.learn()
        if learned is None:
            st.sidebar.caption(
                f"已打开 {read_count} 篇不同文章，其中部分文章已不可用。"
                f"需要至少 {learner.minimum_articles} 篇可读取的文章形成画像，请继续阅读。"
            )
        else:
            persona = learned.persona
            detail = (
                f"聚类画像：{learned.article_count} 篇不同文章聚成 "
                f"{learned.cluster_count} 类，"
                f"最近点开的文章在第 {learned.cluster_index} 类"
                f"（{learned.cluster_size} 篇）；"
                f"代表词：{'、'.join(learned.representative_terms[:4])}"
            )
            if learned.silhouette is not None:
                detail += f"；轮廓系数 {learned.silhouette:.2f}"
            audience = "·".join(
                part
                for part in (
                    learned.persona.education_level,
                    learned.persona.grade,
                    learned.persona.college,
                    learned.persona.major,
                )
                if part and part not in {"不限", "不限专业"}
            )
            detail += (
                f"；受众：{audience}" if audience
                else "；受众：暂未识别（本次仅按主题排序）"
            )
            st.sidebar.caption(detail)
    elif index >= first_persona_index:
        persona = personas[index - first_persona_index]
    if persona is not None:
        st.sidebar.caption(_persona_caption(persona))
        if not tags and persona.goals:
            st.sidebar.caption(
                "提示：把画像目标（" + "、".join(persona.goals)
                + "）加入兴趣标签可同时提升关键词匹配。"
            )
    if persona is not None and not tags and not keywords and persona.id == LEARN_PERSONA_ID:
        st.sidebar.caption("未手动选择标签：已用聚类画像的目标与代表词作为本次查询词。")
        expanded = expand_terms(
            Query(tags=persona.interest_tags or persona.goals,
                  keywords=persona.custom_keywords),
            taxonomy,
        )
    else:
        expanded = expand_terms(Query(tags=tags, keywords=keywords), taxonomy)
    return PersonaSelection(
        persona,
        expanded.all_terms,
        keywords,
        persona.label if persona is not None else "",
        tags,
    )
