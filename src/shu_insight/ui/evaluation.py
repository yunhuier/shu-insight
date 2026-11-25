from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from shu_insight.compliance.detector import sanitize_for_ai
from shu_insight.db.repositories import (
    NoticeRepository,
    RecommendationRunRepository,
    SourceRepository,
)
from shu_insight.evaluation.dataset import (
    DatasetValidationError,
    FrozenDataset,
    load_frozen_dataset,
    split_by_cluster,
    verified_rows_match_repository,
)
from shu_insight.ui.components import render_source_health_card

METRIC_LABELS = {
    "classification_macro_f1": "分类 Macro-F1",
    "entity_f1": "实体抽取 F1",
    "duplicate_f1": "重复识别 Pairwise F1",
    "ndcg_at_10": "排序 NDCG@10",
    "important_notice_miss_rate": "重要通知漏报率",
}

EXPERIMENT_LABELS = {
    "exact_keyword": "Exact Keyword",
    "tfidf": "TF-IDF",
    "semantic": "Semantic",
    "hybrid_final": "Hybrid Final",
    "ablate_persona": "去除画像",
    "ablate_expansion": "去除扩展词",
    "ablate_urgency": "去除紧迫度",
    "ablate_dedup": "去除去重",
    "ai_off": "AI 关闭",
}


def _configured_path(env_name: str, default: Path) -> Path:
    override = os.getenv(env_name, "").strip()
    return Path(override).expanduser().resolve() if override else default


def _load_report(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"评估报告无法读取：{exc}"
    if not isinstance(payload, dict) or not isinstance(payload.get("experiments"), list):
        return None, "评估报告格式无效：缺少 experiments 列表"
    timestamp = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(
        timespec="seconds"
    )
    return payload, timestamp


def _load_dataset(path: Path) -> tuple[FrozenDataset | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        return load_frozen_dataset(path), None
    except DatasetValidationError as exc:
        if "150 至 250" in str(exc) or "缺少 SHA-256 清单" in str(exc):
            try:
                return load_frozen_dataset(path, allow_mini_fixture=True), None
            except (OSError, DatasetValidationError) as mini_exc:
                return None, f"评估数据集无法读取：{mini_exc}"
        return None, f"评估数据集无法读取：{exc}"
    except OSError as exc:
        return None, f"评估数据集无法读取：{exc}"


def _experiment_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in report.get("experiments", []):
        if not isinstance(experiment, dict):
            continue
        metrics = experiment.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        row: dict[str, Any] = {
            "实验": EXPERIMENT_LABELS.get(
                str(experiment.get("name", "")), str(experiment.get("name", ""))
            )
        }
        for key, label in METRIC_LABELS.items():
            value = metrics.get(key)
            row[label] = f"{float(value):.3f}" if isinstance(value, (int, float)) else "—"
        rows.append(row)
    return rows


def _metric(report: dict[str, Any], experiment_name: str, metric: str) -> float | None:
    for experiment in report.get("experiments", []):
        if isinstance(experiment, dict) and experiment.get("name") == experiment_name:
            value = experiment.get("metrics", {}).get(metric)
            return float(value) if isinstance(value, (int, float)) else None
    return None


def _render_dataset_evidence(dataset: FrozenDataset | None) -> None:
    st.subheader("冻结数据集")
    if dataset is None:
        st.info("尚无可读取的冻结评估集；通知中心仍可正常使用。")
        return

    splits = split_by_cluster(
        dataset,
        train=0.70,
        validation=0.15,
        test=0.15,
        seed=42,
    )
    columns = st.columns(4)
    columns[0].metric("总记录", len(dataset.rows))
    columns[1].metric("训练集", len(splits["train"]))
    columns[2].metric("验证集", len(splits["validation"]))
    columns[3].metric("测试集", len(splits["test"]))

    category_counts = Counter(row.category for row in dataset.rows)
    st.caption("类别分布")
    st.bar_chart(
        [{"类别": category, "数量": count} for category, count in category_counts.items()],
        x="类别",
        y="数量",
    )


def _render_cases(
    dataset: FrozenDataset | None,
    report: dict[str, Any] | None,
) -> None:
    st.subheader("错误案例与来源样例（已脱敏）")
    if dataset is None:
        st.caption("冻结评估集可用后，这里会展示经脱敏的代表案例。")
        return

    error_cases = report.get("error_cases", []) if report else []
    if isinstance(error_cases, list) and error_cases:
        st.markdown("**测试集错误案例（由实际算法输出）**")
        rows_by_id = {row.id: row for row in dataset.rows}
        for case in error_cases[:5]:
            if not isinstance(case, dict):
                continue
            row = rows_by_id.get(str(case.get("notice_id", "")))
            title = sanitize_for_ai(row.title) if row else "未知通知"
            kind = sanitize_for_ai(str(case.get("kind", "unknown")))
            expected = sanitize_for_ai(str(case.get("expected", "")))[:120]
            actual = sanitize_for_ai(str(case.get("actual", "")))[:120]
            st.markdown(
                f"- `{kind}` · {title} · 期望：{expected} · 实际：{actual}"
            )

    groups = (
        ("来源成功样例", [row for row in dataset.rows if row.source_success][:2]),
        ("来源失败样例", [row for row in dataset.rows if not row.source_success][:2]),
    )
    for label, rows in groups:
        st.markdown(f"**{label}**")
        if not rows:
            st.caption("暂无样本")
        for row in rows:
            redacted = sanitize_for_ai(row.title)
            st.markdown(f"- {redacted}")


def _render_source_health(
    source_repository: SourceRepository,
    source_names: dict[str, str],
    verified_count_by_source: dict[str, int],
) -> None:
    st.subheader("来源健康状态")
    statuses = {
        status["source_id"]: status for status in source_repository.latest_statuses()
    }
    columns = st.columns(2)
    for index, (source_id, source_name) in enumerate(source_names.items()):
        status = statuses.get(source_id, {})
        with columns[index % 2]:
            render_source_health_card(
                source_name,
                state=status.get("state", "unknown"),
                checked_at=status.get("checked_at"),
                message=status.get("message", ""),
                saved_count=status.get("saved_count", 0),
                updated_count=status.get("updated_count", 0),
                skipped_count=status.get("skipped_count", 0),
                skip_reasons=status.get("skip_reasons", {}),
                visible_count=verified_count_by_source.get(source_id, 0),
            )


def render_evaluation_page(
    *,
    root: Path,
    source_repository: SourceRepository,
    notice_repository: NoticeRepository,
    run_repository: RecommendationRunRepository,
    source_names: dict[str, str],
    verified_count_by_source: dict[str, int] | None = None,
) -> None:
    st.header("实验评估")
    st.caption("本页用于检验通知分类、抽取、去重和推荐效果；只有已核实官网通知与人工复核标注达到正式条件后才显示结果。")

    report_path = _configured_path(
        "SHU_NOTICE_EVALUATION_REPORT_PATH", root / "data" / "evaluation.json"
    )
    dataset_path = _configured_path(
        "SHU_NOTICE_EVALUATION_DATASET_PATH",
        root / "data" / "annotations" / "frozen.jsonl",
    )
    semantic_path = _configured_path(
        "SHU_NOTICE_SEMANTIC_MODEL_PATH",
        root / "data" / "models" / "bge-small-zh-v1.5",
    )
    report, report_timestamp = _load_report(report_path)
    dataset, dataset_error = _load_dataset(dataset_path)
    if dataset_error:
        st.error(dataset_error)
    if (
        report is not None
        and dataset is not None
        and report.get("dataset_sha256") != dataset.sha256
    ):
        st.error("评估报告与当前数据集指纹不一致，暂不显示指标。")

    formal_ready = bool(
        report is not None
        and dataset is not None
        and report.get("formal_metrics_available") is True
        and report.get("corpus_kind") == "verified_official_notices"
        and dataset.corpus_kind == "verified_official_notices"
        and report.get("dataset_sha256") == dataset.sha256
        and verified_rows_match_repository(dataset, notice_repository)
    )
    if not formal_ready:
        st.info("尚未完成真实数据评估；通知中心仍可使用。")
        _render_source_health(source_repository, source_names,
                              verified_count_by_source or {})
        with st.expander("开发者：准备正式评估数据"):
            st.caption(
                "先导出已核实通知，人工标注 150–250 条，再使用 --database 冻结。"
                "迷你夹具仅供算法自检。"
            )
            st.code(
                ".venv\\Scripts\\python.exe scripts\\export_annotation_queue.py "
                "--database data\\shu_insight.sqlite3 "
                "--output data\\local\\annotation-queue.jsonl\n"
                ".venv\\Scripts\\python.exe scripts\\freeze_annotations.py "
                "data\\local\\reviewed.jsonl data\\annotations\\frozen.jsonl "
                "--database data\\shu_insight.sqlite3\n"
                ".venv\\Scripts\\python.exe scripts\\run_evaluation.py "
                "--dataset data\\annotations\\frozen.jsonl --output data\\evaluation.json",
                language="powershell",
            )
        return

    reported_backend = None
    if report is not None:
        hybrid = next(
            (
                item
                for item in report.get("experiments", [])
                if isinstance(item, dict) and item.get("name") == "hybrid_final"
            ),
            None,
        )
        if hybrid is not None:
            reported_backend = hybrid.get("backend")
    backend = str(reported_backend or (
        "semantic（尚未运行报告）" if semantic_path.exists() else "tfidf（语义不可用）"
    ))
    if not semantic_path.exists():
        st.warning("语义模型尚未缓存；当前本机能力显示为 TF-IDF 回退。")

    if report is not None:
        st.markdown("**Exact Keyword** · **TF-IDF** · **Hybrid Final**")
        evidence_columns = st.columns(4)
        evidence_columns[0].metric("报告时间", report_timestamp or "未知")
        evidence_columns[1].metric("数据集记录", int(report.get("row_count", 0)))
        evidence_columns[2].metric("运行后端", backend)
        evidence_columns[3].metric("推荐运行", run_repository.summary()["run_count"])
        st.markdown(f"数据集 SHA-256：`{report.get('dataset_sha256', '未知')}`")
        table_rows = _experiment_rows(report)
        if table_rows:
            st.dataframe(table_rows, hide_index=True, width="stretch")

        final_ndcg = _metric(report, "hybrid_final", "ndcg_at_10")
        final_miss = _metric(report, "hybrid_final", "important_notice_miss_rate")
        metric_columns = st.columns(2)
        metric_columns[0].metric(
            "Hybrid Final NDCG@10",
            f"{final_ndcg:.3f}" if final_ndcg is not None else "—",
        )
        metric_columns[1].metric(
            "重要通知漏报率（越低越好）",
            f"{final_miss:.3f}" if final_miss is not None else "—",
        )

        ablation_rows = [
            {
                "实验": EXPERIMENT_LABELS.get(str(item.get("name")), str(item.get("name"))),
                "NDCG@10": float(item.get("metrics", {}).get("ndcg_at_10", 0.0)),
            }
            for item in report.get("experiments", [])
            if isinstance(item, dict)
            and (
                item.get("name") == "hybrid_final"
                or str(item.get("name", "")).startswith("ablate_")
            )
        ]
        if ablation_rows:
            st.subheader("消融实验")
            st.bar_chart(ablation_rows, x="实验", y="NDCG@10")

    _render_dataset_evidence(dataset)
    _render_cases(dataset, report)
    _render_source_health(source_repository, source_names,
                          verified_count_by_source or {})
