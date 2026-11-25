from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

import streamlit as st

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import (
    ArticleReadRepository,
    ManualUpdateRunRepository,
    NoticeRepository,
    PersonaRepository,
)
from shu_insight.db.schema import Database
from shu_insight.domain.models import ParseState
from shu_insight.ingest.http_client import HttpCache, PoliteHttpClient
from shu_insight.processing.classify import load_taxonomy
from shu_insight.profiles.learning import ArticleReadLearner
from shu_insight.profiles.service import DEMO_PERSONAS, PersonaService
from shu_insight.ranking.score import load_ranking_weights
from shu_insight.services.notices import NoticeService
from shu_insight.services.update import UpdateService
from shu_insight.settings import AppSettings
from shu_insight.sources.registry import load_source_registry
from shu_insight.ui.components import (
    manual_update_progress_text,
    render_manual_run_status,
    render_manual_update_result,
    stale_source_ids,
)
from shu_insight.ui.home import render_home
from shu_insight.ui.personas import render_interest_panel
from shu_insight.ui.update_progress import ManualUpdateProgress

PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="上大知讯", page_icon="🔎", layout="wide")
st.title("🔎 上大知讯")
st.markdown("**非上海大学官方产品，信息以官网原文为准。**")
st.caption("选择兴趣标签发现校内通知，或用关键词搜索已保存的官网信息。")

settings = AppSettings.load(PROJECT_ROOT)
database = Database(settings.database_path)
database.initialize()
registry = load_source_registry(settings.config_dir / "sources.toml")
taxonomy = load_taxonomy(settings.config_dir / "taxonomy.toml")
policy = OfficialNoticePolicy({key: adapter.config for key, adapter in registry.items()})
notice_repository = NoticeRepository(database, policy=policy)
notice_service = NoticeService(
    notice_repository,
    authority_by_source_id={
        source_id: adapter.config.authority_score for source_id, adapter in registry.items()
    },
    ranking_weights=load_ranking_weights(settings.config_dir / "ranking.toml"),
    taxonomy=taxonomy,
)
saved_personas = PersonaService(PersonaRepository(database)).list_all()
personas = saved_personas or list(DEMO_PERSONAS)
learner = ArticleReadLearner(
    ArticleReadRepository(database),
    taxonomy,
    notice_source=notice_service.all,
)
selection = render_interest_panel(taxonomy, personas, learner=learner)
st.sidebar.header("更新通知")
st.sidebar.caption("仅点击时检查官网和本科生院；其余来源使用本地存量。")
if st.sidebar.button("更新官网和本科生院"):
    source_names = {key: adapter.config.name for key, adapter in registry.items()
                    if key in UpdateService.MANUAL_SOURCE_IDS}
    progress_state = ManualUpdateProgress(source_names)
    update_status = st.status("准备检查官网和本科生院…", expanded=True)
    progress_bar = update_status.progress(0.0, text=progress_state.progress_text)
    progress_detail = update_status.empty()

    def show_progress(event):
        progress_state.observe(event)
        update_status.update(label=progress_state.label)
        progress_bar.progress(progress_state.fraction, text=progress_state.progress_text)
        text = manual_update_progress_text(event, source_names)
        if text is not None:
            progress_detail.caption(text)

    def create_client():
        return PoliteHttpClient(
            contact=settings.effective_contact(settings.contact),
            cache=HttpCache(settings.database_path.parent / "http-validators.sqlite3"),
        )

    update_report = UpdateService(
        database=database, registry=registry, client_factory=create_client,
        taxonomy=taxonomy,
    ).update_all(on_event=show_progress)
    st.session_state["manual-update-report"] = update_report
    progress_state.finish(update_report)
    progress_bar.progress(progress_state.fraction, text=progress_state.progress_text)
    update_status.update(label=progress_state.label, state=progress_state.status_state)

report = st.session_state.get("manual-update-report")
if report is not None:
    render_manual_update_result(report)
notices = notice_service.all()
source_counts = Counter(notice.source_id for notice in notices)
st.sidebar.header("收录统计")
st.sidebar.caption("当前本地存量，不受标签或搜索影响。")
for source_id, source_label in (
    ("shu_main", "上海大学官网"), ("undergraduate", "本科生院"),
    ("graduate", "研究生院"), ("computer", "计算机学院"),
):
    st.sidebar.markdown(f"{source_label}：**{source_counts[source_id]} 条**")
st.sidebar.caption(f"合计 {len(notices)} 条，包含仅列表可见的通知。")
parsed_count = sum(notice.parse_state is not ParseState.LISTING_ONLY for notice in notices)
st.caption(f"本地可浏览 {len(notices)} 条 · 正文已读取 {parsed_count} 条 · "
           f"仅列表可见 {len(notices) - parsed_count} 条")
st.caption("本地存量覆盖上海大学官网、本科生院、研究生院、计算机工程与科学学院；"
           "列表卡片仅依据官网标题，正文内容以原文为准。")
render_home(notice_service, selection, learner=learner)
with st.expander("本地更新记录与规则"):
    render_manual_run_status(ManualUpdateRunRepository(database).latest())
    st.caption("手动更新仅访问上海大学官网、本科生院。每站最多 5 个已审阅列表页、"
               "40 次物理请求，同主机相邻尝试至少间隔 2 秒。")
    stale = stale_source_ids(registry, date.today())
    if stale:
        st.caption("来源配置需重新审阅：" + "、".join(stale))
