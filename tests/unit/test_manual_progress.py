from shu_insight.ingest.pipeline import SourceUpdateResult, UpdateEvent, UpdateReport


def progress():
    from shu_insight.ui.update_progress import ManualUpdateProgress

    return ManualUpdateProgress({"shu_main": "上海大学官网", "undergraduate": "本科生院"})


def test_counts_real_events_without_guessing_article_percentage():
    state = progress()
    assert state.fraction == 0
    state.observe(UpdateEvent("source_started", "shu_main"))
    state.observe(UpdateEvent("list_page_read", "shu_main"))
    state.observe(UpdateEvent("notice_saved", "shu_main"))
    state.observe(UpdateEvent("notice_skipped", "shu_main"))
    assert state.fraction == 0
    assert "上海大学官网" in state.label
    assert (state.pages_read, state.written, state.skipped) == (1, 1, 1)
    state.observe(UpdateEvent("source_finished", "shu_main"))
    state.observe(UpdateEvent("source_finished", "shu_main"))
    assert state.fraction == 0.5


def test_failure_ends_step_without_claiming_success():
    state = progress()
    state.observe(UpdateEvent("source_failed", "shu_main", "connection_error"))
    assert state.fraction == 0.5
    assert "停止" in state.label
    state.observe(UpdateEvent("source_started", "undergraduate"))
    state.finish(UpdateReport({
        "shu_main": SourceUpdateResult("error"),
        "undergraduate": SourceUpdateResult("partial", saved_count=3),
    }, (), state="partial"))
    assert state.fraction == 1.0 and state.written == 3
    assert state.status_state == "complete"
    assert "部分" in state.label and "成功" not in state.label


def test_empty_failure_and_unknown_sources_cannot_invent_progress():
    state = progress()
    state.observe(UpdateEvent("notice_saved", "graduate"))
    state.observe(UpdateEvent("source_finished", "graduate"))
    assert state.fraction == 0 and state.written == 0
    state.finish(UpdateReport({}, (), state="failed"))
    assert state.fraction == 0 and state.status_state == "error"
    assert "未能" in state.label


def test_next_source_start_closes_previous_partial_step():
    state = progress()
    state.observe(UpdateEvent("source_started", "shu_main"))
    state.observe(UpdateEvent("source_started", "undergraduate"))
    assert state.fraction == 0.5
    assert "1/2" in state.progress_text
    state.finish(UpdateReport({}, (), state="concurrent_skipped"))
    assert "正在进行" in state.label
