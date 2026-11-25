from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest

from shu_insight.compliance.official import OfficialNoticePolicy
from shu_insight.db.repositories import NoticeRepository
from shu_insight.db.schema import Database
from shu_insight.evaluation.dataset import (
    AnnotationRow,
    DatasetValidationError,
    FrozenDataset,
    freeze_annotations,
    load_frozen_dataset,
    split_by_cluster,
    verified_rows_match_repository,
)
from shu_insight.ingest.snapshot import import_snapshot
from shu_insight.sources.registry import load_source_registry

PERSONA_IDS = (
    "demo-undergrad-research",
    "demo-undergrad-job",
    "demo-master",
    "demo-freshman",
)


def make_rows(count: int, *, reviewed_by: str) -> list[dict[str, object]]:
    return [
        {
            "id": f"row-{index:03d}",
            "source_id": "computer" if index % 2 else "shu_main",
            "title": f"人工智能竞赛通知 {index}",
            "context_excerpt": "面向本科生的竞赛报名安排",
            "original_url": f"https://example.edu/info/{index}.htm",
            "category": "竞赛创新",
            "gold_entities": ["本科生"],
            "extracted_entities": ["本科生"],
            "duplicate_cluster_id": f"cluster-{index // 3}",
            "predicted_duplicate_cluster_id": f"cluster-{index // 3}",
            "relevance_labels": {persona_id: index % 4 for persona_id in PERSONA_IDS},
            "important": index % 7 == 0,
            "source_success": True,
            "reviewed_by": reviewed_by,
            "evidence_corrections": {"category": "confirmed"},
        }
        for index in range(count)
    ]


def _verified_repository(tmp_path: Path) -> NoticeRepository:
    database = Database(tmp_path / "verified.sqlite3")
    database.initialize()
    registry = load_source_registry(Path("config/sources.toml"))
    policy = OfficialNoticePolicy({key: value.config for key, value in registry.items()})
    repository = NoticeRepository(database, policy=policy)
    import_snapshot(Path("data/official/snapshot-2026-09-24.json"), repository)
    return repository


def test_forged_https_rows_cannot_be_formally_frozen(tmp_path: Path) -> None:
    repository = _verified_repository(tmp_path)
    output = tmp_path / "frozen.jsonl"
    with pytest.raises(DatasetValidationError, match="已核实"):
        freeze_annotations(make_rows(150, reviewed_by="human"), output,
                           verified_repository=repository)
    assert not output.exists()
    assert not Path(f"{output}.provenance.json").exists()


def test_forged_title_cannot_be_formally_frozen(tmp_path: Path) -> None:
    repository = _verified_repository(tmp_path)
    official = repository.list_verified()[0]
    rows = make_rows(150, reviewed_by="human")
    rows[0].update(id=official.id, source_id=official.source_id,
                   original_url=official.canonical_url, title="虚构的通知标题")
    with pytest.raises(DatasetValidationError, match="标题"):
        freeze_annotations(rows, tmp_path / "frozen.jsonl",
                           verified_repository=repository)


def test_invalid_formal_sidecar_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "frozen.jsonl"
    freeze_annotations(make_rows(150, reviewed_by="human"), output)
    Path(f"{output}.provenance.json").write_text(
        json.dumps({"corpus_kind": "verified_official_notices", "dataset_sha256": "wrong"}),
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="证明"):
        load_frozen_dataset(output)
    assert load_frozen_dataset(output, allow_mini_fixture=True).corpus_kind == "self_test"


def test_current_repository_revocation_and_title_change_invalidate_rows(tmp_path: Path) -> None:
    repository = _verified_repository(tmp_path)
    notice = repository.list_verified()[0]
    row = AnnotationRow.from_mapping({
        **make_rows(1, reviewed_by="human")[0],
        "id": notice.id,
        "source_id": notice.source_id,
        "original_url": notice.canonical_url,
        "title": notice.title,
    })
    proof_row = [notice.id, notice.canonical_url, notice.proof_status.value,
                 notice.verified_at.isoformat(), notice.verification_ref]
    proof_digest = hashlib.sha256(json.dumps(
        [proof_row], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    dataset = FrozenDataset((row,), "audited-sample", tmp_path / "sample.jsonl",
                            "verified_official_notices", proof_digest)
    assert verified_rows_match_repository(dataset, repository)

    from dataclasses import replace

    assert not verified_rows_match_repository(
        replace(dataset, rows=(replace(row, title="虚构通知标题"),)), repository
    )
    with repository.database.transaction() as connection:
        connection.execute("UPDATE notices SET proof_status = ? WHERE id = ?",
                           ("unverified_legacy", notice.id))
    assert not verified_rows_match_repository(dataset, repository)


def test_freeze_requires_150_to_250_human_reviewed_rows(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="150 至 250"):
        freeze_annotations(make_rows(149, reviewed_by="human"), tmp_path / "frozen.jsonl")
    with pytest.raises(DatasetValidationError, match="人工复核"):
        freeze_annotations(make_rows(150, reviewed_by="ai"), tmp_path / "frozen.jsonl")


def test_freeze_writes_hash_and_detects_later_tampering(tmp_path: Path) -> None:
    output = tmp_path / "frozen.jsonl"
    frozen = freeze_annotations(make_rows(150, reviewed_by="human"), output)

    loaded = load_frozen_dataset(output)
    assert loaded.sha256 == frozen.sha256
    assert len(loaded.rows) == 150
    assert loaded.corpus_kind == "self_test"
    assert loaded.proof_digest is None

    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(make_rows(1, reviewed_by="human")[0]) + "\n")
    with pytest.raises(DatasetValidationError, match="SHA-256"):
        load_frozen_dataset(output)


def test_duplicate_cluster_never_crosses_splits(tmp_path: Path) -> None:
    dataset = freeze_annotations(
        make_rows(150, reviewed_by="human"),
        tmp_path / "frozen.jsonl",
    )

    split = split_by_cluster(dataset, train=0.6, validation=0.2, test=0.2, seed=20260921)
    locations: dict[str, set[str]] = defaultdict(set)
    for partition, rows in split.items():
        for row in rows:
            locations[row.duplicate_cluster_id].add(partition)

    assert all(len(parts) == 1 for parts in locations.values())
    assert sum(len(rows) for rows in split.values()) == 150


def test_three_cluster_self_test_populates_each_requested_partition() -> None:
    fixture = load_frozen_dataset(
        Path("tests/fixtures/annotations/mini_frozen_set.jsonl"),
        allow_mini_fixture=True,
    )

    split = split_by_cluster(
        fixture, train=0.70, validation=0.15, test=0.15, seed=42
    )

    assert all(split[partition] for partition in ("train", "validation", "test"))
