import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.export_annotation_queue import export_annotation_queue
from scripts.run_evaluation import main as run_evaluation_main
from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository
from shu_insight.db.schema import Database
from shu_insight.domain.models import DiscoveredNotice, NoticeDraft, ProofStatus, Verification
from shu_insight.evaluation.dataset import (
    FrozenDataset,
    load_frozen_dataset,
    split_by_cluster,
)
from shu_insight.evaluation.runner import ExperimentRunner
from shu_insight.processing.classify import load_taxonomy
from shu_insight.ranking.expand import Query, expand_terms
from shu_insight.ranking.relevance import TfidfBackend
from shu_insight.sources.registry import load_source_registry


def dataset():
    mini = load_frozen_dataset(Path("tests/fixtures/annotations/mini_frozen_set.jsonl"),
                               allow_mini_fixture=True)
    rows = []
    for number in range(20):
        for suffix in ("a", "b"):
            rows.append(replace(
                mini.rows[0], id=f"{number}-{suffix}", duplicate_cluster_id=f"gold-{number}",
                original_url=f"https://demo.invalid/{number}/{suffix}",
                title=f"科研项目{number}报名通知",
                context_excerpt="面向本科生。报名截止：2026年9月23日18:00。",
                category="错误金标准", extracted_entities=("伪造预测",),
                predicted_duplicate_cluster_id=f"lie-{number}-{suffix}",
            ))
    return FrozenDataset(tuple(rows), "self-test", Path("synthetic.jsonl"))


def test_annotation_export_uses_parsed_proof_when_list_card_is_browseable(tmp_path):
    database = Database(tmp_path / "evaluation.sqlite3")
    database.initialize()
    adapter = load_source_registry(Path("config/sources.toml"))["graduate"]
    repository = NoticeRepository(
        database, policy=OfficialNoticePolicy({"graduate": adapter.config}),
    )
    checked_at = datetime(2026, 10, 7, tzinfo=UTC)
    parsed_url = "https://gs.shu.edu.cn/info/1029/201.htm"
    listing_url = "https://gs.shu.edu.cn/info/1029/202.htm"
    parsed = repository.upsert(
        NoticeDraft(
            source_id="graduate", canonical_url=parsed_url,
            title="研究生课程申请通知",
            normalized_text="研究生按照培养计划提交课程申请并由学院审核。",
        ),
        verification=Verification(ProofStatus.VERIFIED_LIVE, checked_at, "live:fixture"),
    )
    listing = repository.upsert_listing(
        DiscoveredNotice("graduate", listing_url, "研究生课程安排通知"),
        list_url=adapter.config.list_url, checked_at=checked_at,
    )

    output = tmp_path / "annotation-queue.jsonl"
    assert len(repository.list_browseable()) == 2
    assert export_annotation_queue(repository, output) == 1
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [(row["id"], row["original_url"]) for row in rows] == [
        (parsed.notice_id, parsed_url),
    ]
    assert listing.notice_id not in {row["id"] for row in rows}
    assert repository.detail_evidence_verified(listing.notice_id) == {
        "duplicates": [], "possible_duplicates": [], "versions": [],
    }


def test_evaluation_uses_heldout_algorithms_and_does_not_trust_precomputed_predictions():
    data = dataset()
    runner = ExperimentRunner(data, now=datetime(2026, 9, 22, tzinfo=UTC))
    report = runner.run().to_mapping()
    assert report["evidence_kind"] == "synthetic_self_test"
    assert report["formal_metrics_available"] is False
    assert report["corpus_kind"] == "self_test"
    splits = report["split_ids"]
    assert all(splits[part] for part in ("train", "validation", "test"))
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert len(splits["test"]) < len(data.rows)
    experiments = {item["name"]: item for item in report["experiments"]}
    assert experiments["semantic"]["status"] == "unavailable"
    assert experiments["semantic"]["metrics"] == {}
    assert experiments["ai_off"]["status"] == "not_applicable"
    assert experiments["tfidf"]["fitted_ids"] == splits["train"]
    assert experiments["hybrid_final"]["metrics"]["entity_f1"] == 1.0
    assert experiments["hybrid_final"]["metrics"]["duplicate_f1"] == 1.0
    assert experiments["hybrid_final"]["metrics"]["recall_at_10"] == 1.0
    assert experiments["hybrid_final"]["metrics"]["important_notice_miss_rate"] == 0.0
    assert report["error_cases"] and all(
        case["notice_id"] in splits["test"] for case in report["error_cases"]
    )
    changed = replace(data, rows=tuple(replace(row, extracted_entities=(),
                       predicted_duplicate_cluster_id="all-one") for row in data.rows))
    assert ExperimentRunner(changed, now=runner.now).run().to_mapping() == report


def test_tfidf_scores_are_actual_train_fitted_cosines_and_ablations_change_pipeline():
    data = dataset()
    report = ExperimentRunner(data, now=datetime(2026, 9, 22, tzinfo=UTC)).run().to_mapping()
    splits = report["split_ids"]
    by_id = {row.id: row for row in data.rows}
    backend = TfidfBackend()
    backend.fit([f"{by_id[key].title}\n{by_id[key].context_excerpt}" for key in splits["train"]])
    query = expand_terms(Query(tags=("保研", "科研")), load_taxonomy(Path("config/taxonomy.toml")))
    results = {item["name"]: item for item in report["experiments"]}
    first = results["tfidf"]["rankings"]["demo-undergrad-research"][0]
    row = by_id[first["notice_id"]]
    assert first["score"]["relevance"] == pytest.approx(
        backend.score(query, f"{row.title}\n{row.context_excerpt}").score
    )
    def rank(name):
        return results[name]["rankings"]["demo-undergrad-research"]

    assert len(rank("hybrid_final")) < len(rank("ablate_dedup"))

    def scores(name):
        return {
            item["notice_id"]: item["score"]["total"] for item in rank(name)
        }

    assert scores("hybrid_final") != scores("ablate_urgency")
    assert scores("hybrid_final") != scores("ablate_persona")
    assert scores("hybrid_final") != scores("ablate_expansion")


def test_ranked_low_relevance_version_does_not_inherit_cluster_max_label():
    data = dataset()
    test_rows = split_by_cluster(
        data, train=0.70, validation=0.15, test=0.15, seed=42
    )["test"]
    low_version_id = min(row.id for row in test_rows)
    changed = replace(
        data,
        rows=tuple(
            replace(
                row,
                relevance_labels={
                    **row.relevance_labels,
                    "demo-undergrad-research": 0,
                },
            )
            if row.id == low_version_id
            else row
            for row in data.rows
        ),
    )

    report = ExperimentRunner(
        changed, now=datetime(2026, 9, 22, tzinfo=UTC)
    ).run().to_mapping()
    hybrid = next(item for item in report["experiments"] if item["name"] == "hybrid_final")
    entries = hybrid["rankings"]["demo-undergrad-research"]
    selected = next(item for item in entries if item["notice_id"] == low_version_id)

    assert selected["label"] == 0
    assert hybrid["metrics"]["precision_at_10"] < 1.0


def test_cli_labels_mini_report_as_self_test(tmp_path, monkeypatch, capsys):
    output = tmp_path / "mini-report.json"
    monkeypatch.setattr(sys, "argv", [
        "run_evaluation.py", "--allow-mini-fixture", "--dataset",
        "tests/fixtures/annotations/mini_frozen_set.jsonl", "--output", str(output),
    ])
    assert run_evaluation_main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["corpus_kind"] == "self_test"
    assert report["formal_metrics_available"] is False
    assert "self_test" in capsys.readouterr().out
