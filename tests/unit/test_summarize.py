from shu_insight.processing.summarize import summarize_article


def test_summary_keeps_action_sentence_after_generic_introduction() -> None:
    body = "为贯彻相关工作精神，现将有关事项通知如下。" * 25 + "本科生请于9月30日前完成报名。"
    summary = summarize_article(body)
    assert "本科生请于9月30日前完成报名。" in summary
    assert summary.endswith("。") and len(summary) <= 350


def test_long_sentence_produces_a_bounded_preview_instead_of_empty_summary() -> None:
    summary = summarize_article("本科生请完成报名" + "相关事项" * 100, limit=350)
    assert summary.startswith("本科生请完成报名")
    assert len(summary) <= 350 and summary.endswith("…")
