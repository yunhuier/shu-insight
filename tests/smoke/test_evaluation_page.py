import hashlib
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository, SourceRepository
from shu_insight.db.schema import Database
from shu_insight.evaluation.dataset import AnnotationRow, FrozenDataset, load_frozen_dataset
from shu_insight.ingest.snapshot import import_snapshot
from shu_insight.sources.registry import load_source_registry
from shu_insight.ui import evaluation as evaluation_ui

APP_PATH = Path(__file__).resolve().parents[2] / "app.py"
MINI_DATASET = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "annotations"
    / "mini_frozen_set.jsonl"
)


def _evaluation_harness(project_root: str) -> None:
    """Exercise the development renderer without exposing it in the main app."""
    from collections import Counter
    from pathlib import Path

    from shu_insight.compliance.official import OfficialNoticePolicy
    from shu_insight.db.repositories import (
        NoticeRepository,
        RecommendationRunRepository,
        SourceRepository,
    )
    from shu_insight.db.schema import Database
    from shu_insight.settings import AppSettings
    from shu_insight.sources.registry import load_source_registry
    from shu_insight.ui.evaluation import render_evaluation_page

    root = Path(project_root)
    settings = AppSettings.load(root)
    database = Database(settings.database_path)
    database.initialize()
    registry = load_source_registry(settings.config_dir / 'sources.toml')
    repository = NoticeRepository(database, OfficialNoticePolicy({
        key: adapter.config for key, adapter in registry.items()
    }))
    render_evaluation_page(
        root=root, source_repository=SourceRepository(database), notice_repository=repository,
        run_repository=RecommendationRunRepository(database),
        source_names={key: adapter.config.name for key, adapter in registry.items()},
        verified_count_by_source=Counter(notice.source_id for notice in repository.list_verified()),
    )


def _evaluation_app() -> AppTest:
    return AppTest.from_function(_evaluation_harness, args=(str(APP_PATH.parent),),
                                 default_timeout=15)


def _write_report(path: Path, dataset_sha256: str) -> None:
    metrics = {
        "classification_macro_f1": 0.75,
        "entity_f1": 0.80,
        "duplicate_f1": 1.0,
        "ndcg_at_10": 0.70,
        "important_notice_miss_rate": 0.10,
    }
    path.write_text(
        json.dumps(
            {
                "dataset_sha256": dataset_sha256,
                "row_count": 4,
                "evidence_kind": "synthetic_self_test",
                "formal_metrics_available": False,
                "error_cases": [
                    {
                        "notice_id": "mini-4",
                        "kind": "classification",
                        "expected": "就业实习",
                        "actual": "校园事务",
                    }
                ],
                "experiments": [
                    {"name": "exact_keyword", "metrics": metrics},
                    {"name": "tfidf", "metrics": {**metrics, "ndcg_at_10": 0.76}},
                    {
                        "name": "hybrid_final",
                        "metrics": {**metrics, "ndcg_at_10": 0.88},
                    },
                    {
                        "name": "ablate_persona",
                        "metrics": {**metrics, "ndcg_at_10": 0.79},
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_dataset_with_sensitive_failure(path: Path) -> None:
    rows = [json.loads(line) for line in MINI_DATASET.read_text(encoding="utf-8").splitlines()]
    rows[-1]["context_excerpt"] = "抓取失败，联系人手机 13812345678"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def app_test(tmp_path: Path, monkeypatch) -> AppTest:
    report_path = tmp_path / "evaluation.json"
    dataset_path = tmp_path / "frozen.jsonl"
    _write_dataset_with_sensitive_failure(dataset_path)
    _write_report(
        report_path,
        load_frozen_dataset(dataset_path, allow_mini_fixture=True).sha256,
    )
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(tmp_path / "app.sqlite3"))
    monkeypatch.setenv("SHU_NOTICE_EVALUATION_REPORT_PATH", str(report_path))
    monkeypatch.setenv("SHU_NOTICE_EVALUATION_DATASET_PATH", str(dataset_path))
    monkeypatch.setenv("SHU_NOTICE_SEMANTIC_MODEL_PATH", str(tmp_path / "missing-model"))
    monkeypatch.delenv("SHU_NOTICE_CONTACT", raising=False)
    return _evaluation_app()


def _open_evaluation(app_test: AppTest) -> AppTest:
    return app_test.run()


def test_mini_fixture_never_renders_scores_or_cases(
    app_test: AppTest,
) -> None:
    at = _open_evaluation(app_test)
    rendered = " ".join(
        str(node.value) for kind in ("markdown", "caption", "metric", "info")
        for node in at.get(kind)
    )

    assert not at.exception
    assert "尚未完成真实数据评估" in rendered
    assert "mini-4" not in rendered
    assert "classification" not in rendered
    assert not any(metric.label == "Hybrid Final NDCG@10" for metric in at.metric)


def test_report_with_different_dataset_hash_does_not_show_scores(
    app_test: AppTest,
    tmp_path: Path,
) -> None:
    _write_report(tmp_path / "evaluation.json", "wrong-fingerprint")

    at = _open_evaluation(app_test)

    assert any("指纹不一致" in error.value for error in at.error)
    assert not any(metric.label == "Hybrid Final NDCG@10" for metric in at.metric)


def test_failure_examples_are_hidden_without_formal_provenance(app_test: AppTest) -> None:
    at = _open_evaluation(app_test)
    rendered = " ".join(
        str(node.value)
        for element_type in ("markdown", "caption", "info", "warning", "text")
        for node in at.get(element_type)
    )

    assert not at.exception
    assert "13812345678" not in rendered
    assert "[REDACTED_PHONE]" not in rendered


def test_revoked_official_proof_hides_claimed_formal_report(
    tmp_path: Path, monkeypatch,
) -> None:
    database_path = tmp_path / "verified.sqlite3"
    database = Database(database_path)
    database.initialize()
    registry = load_source_registry(Path("config/sources.toml"))
    policy = OfficialNoticePolicy({key: item.config for key, item in registry.items()})
    repository = NoticeRepository(database, policy=policy)
    import_snapshot(Path("data/official/snapshot-2026-09-24.json"), repository)
    notice = repository.list_verified()[0]
    sample = json.loads(MINI_DATASET.read_text(encoding="utf-8").splitlines()[0])
    sample.update(id=notice.id, source_id=notice.source_id,
                  original_url=notice.canonical_url, title=notice.title)
    row = AnnotationRow.from_mapping(sample)
    proof = [[notice.id, notice.canonical_url, notice.proof_status.value,
              notice.verified_at.isoformat(), notice.verification_ref]]
    digest = hashlib.sha256(json.dumps(
        proof, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    dataset = FrozenDataset((row,), "sample-sha", tmp_path / "sample.jsonl",
                            "verified_official_notices", digest)
    monkeypatch.setattr(evaluation_ui, "_load_dataset", lambda _path: (dataset, None))
    report_path = tmp_path / "claimed-formal.json"
    report_path.write_text(json.dumps({
        "dataset_sha256": dataset.sha256,
        "corpus_kind": "verified_official_notices",
        "formal_metrics_available": True,
        "row_count": 1,
        "experiments": [{"name": "hybrid_final", "metrics": {"ndcg_at_10": 0.99}}],
        "error_cases": [{"notice_id": notice.id, "kind": "classification",
                         "expected": "official", "actual": "wrong"}],
    }), encoding="utf-8")
    with database.transaction() as connection:
        connection.execute("UPDATE notices SET proof_status = ? WHERE id = ?",
                           ("unverified_legacy", notice.id))
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("SHU_NOTICE_EVALUATION_REPORT_PATH", str(report_path))

    at = _open_evaluation(_evaluation_app())
    rendered = " ".join(str(node.value) for node in at.markdown)
    assert not at.exception
    assert any("尚未完成真实数据评估" in info.value for info in at.info)
    assert notice.title not in rendered
    assert not any(metric.label == "Hybrid Final NDCG@10" for metric in at.metric)


def test_missing_assets_and_source_failure_keep_page_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "missing-state.sqlite3"
    database = Database(database_path)
    database.initialize()
    SourceRepository(database).record_status(
        {
            "source_id": "computer",
            "source_name": "计算机工程与科学学院",
            "state": "error",
            "message": "fixture failure",
        }
    )
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(database_path))
    monkeypatch.setenv(
        "SHU_NOTICE_EVALUATION_REPORT_PATH", str(tmp_path / "missing-report.json")
    )
    monkeypatch.setenv(
        "SHU_NOTICE_EVALUATION_DATASET_PATH", str(tmp_path / "missing-dataset.jsonl")
    )
    monkeypatch.setenv("SHU_NOTICE_SEMANTIC_MODEL_PATH", str(tmp_path / "missing-model"))

    at = _open_evaluation(_evaluation_app())

    assert not at.exception
    assert any("尚未完成真实数据评估" in info.value for info in at.info)
    assert any(metric.value == "异常" for metric in at.metric)
    assert any("run_evaluation.py" in code.value for code in at.code)


def test_source_health_distinguishes_successful_zero_result_and_skip_reasons(
    tmp_path: Path, monkeypatch,
) -> None:
    path = tmp_path / "zero-result.sqlite3"
    database = Database(path)
    database.initialize()
    SourceRepository(database).record_status({
        "source_id": "computer", "state": "success", "saved_count": 0,
        "updated_count": 0, "skipped_count": 2,
        "skip_reasons": {"sensitive_content": 2},
    })
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(path))
    at = _open_evaluation(_evaluation_app())
    captions = " ".join(item.value for item in at.caption)
    assert not at.exception
    assert "访问成功，0 条已核实" in captions
    assert "跳过：2" in captions and "sensitive_content 2" in captions


def test_offline_seed_source_health_says_not_checked(tmp_path: Path, monkeypatch) -> None:
    from scripts.seed_demo import seed_demo

    path = tmp_path / "offline.sqlite3"
    seed_demo(path, reset=True)
    monkeypatch.setenv("SHU_NOTICE_DATABASE_PATH", str(path))
    at = _open_evaluation(_evaluation_app())
    rendered = " ".join(str(item.value) for kind in ("caption", "metric")
                        for item in at.get(kind))
    assert not at.exception
    assert "尚未检查官网" in rendered
    assert "虚构演示" not in rendered
    assert "访问成功" not in rendered
