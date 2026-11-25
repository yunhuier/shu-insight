from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shu_insight.db.repositories import NoticeRepository
from shu_insight.domain.models import NoticeRecord

REQUIRED_PERSONA_IDS = frozenset(
    {
        "demo-undergrad-research",
        "demo-undergrad-job",
        "demo-master",
        "demo-freshman",
    }
)


class DatasetValidationError(ValueError):
    """Raised when annotations cannot be treated as a frozen evaluation set."""


@dataclass(frozen=True, slots=True)
class AnnotationRow:
    id: str
    source_id: str
    title: str
    context_excerpt: str
    original_url: str
    category: str
    gold_entities: tuple[str, ...]
    extracted_entities: tuple[str, ...]
    duplicate_cluster_id: str
    predicted_duplicate_cluster_id: str
    relevance_labels: dict[str, int]
    important: bool
    source_success: bool
    reviewed_by: str
    evidence_corrections: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AnnotationRow:
        try:
            return cls(
                id=str(value["id"]),
                source_id=str(value["source_id"]),
                title=str(value["title"]),
                context_excerpt=str(value.get("context_excerpt", ""))[:500],
                original_url=str(value["original_url"]),
                category=str(value["category"]),
                gold_entities=tuple(map(str, value.get("gold_entities", ()))),
                extracted_entities=tuple(map(str, value.get("extracted_entities", ()))),
                duplicate_cluster_id=str(value["duplicate_cluster_id"]),
                predicted_duplicate_cluster_id=str(
                    value.get("predicted_duplicate_cluster_id", value["duplicate_cluster_id"])
                ),
                relevance_labels={
                    str(key): int(label)
                    for key, label in dict(value["relevance_labels"]).items()
                },
                important=bool(value.get("important", False)),
                source_success=bool(value.get("source_success", True)),
                reviewed_by=str(value["reviewed_by"]),
                evidence_corrections=dict(value["evidence_corrections"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetValidationError(f"invalid annotation row: {exc}") from exc

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "title": self.title,
            "context_excerpt": self.context_excerpt,
            "original_url": self.original_url,
            "category": self.category,
            "gold_entities": self.gold_entities,
            "extracted_entities": self.extracted_entities,
            "duplicate_cluster_id": self.duplicate_cluster_id,
            "predicted_duplicate_cluster_id": self.predicted_duplicate_cluster_id,
            "relevance_labels": self.relevance_labels,
            "important": self.important,
            "source_success": self.source_success,
            "reviewed_by": self.reviewed_by,
            "evidence_corrections": self.evidence_corrections,
        }


@dataclass(frozen=True, slots=True)
class FrozenDataset:
    rows: tuple[AnnotationRow, ...]
    sha256: str
    path: Path
    corpus_kind: str = "self_test"
    proof_digest: str | None = None


def _manifest_path(path: Path) -> Path:
    return Path(f"{path}.sha256")


def _provenance_path(path: Path) -> Path:
    return Path(f"{path}.provenance.json")


def _proof_digest(notices: Iterable[NoticeRecord]) -> str:
    proof_rows = [
        (
            notice.id,
            notice.canonical_url,
            notice.proof_status.value,
            notice.verified_at.isoformat(),
            notice.verification_ref,
        )
        for notice in notices
        if notice.verified_at is not None
    ]
    payload = json.dumps(sorted(proof_rows), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _matching_verified_notices(
    rows: tuple[AnnotationRow, ...], repository: NoticeRepository
) -> list[NoticeRecord] | None:
    verified = {notice.id: notice for notice in repository.list_verified()}
    matched: list[NoticeRecord] = []
    for row in rows:
        notice = verified.get(row.id)
        if notice is None or (notice.source_id, notice.canonical_url) != (
            row.source_id, row.original_url
        ):
            return None
        if notice.title != row.title:
            raise DatasetValidationError("标注标题与已核实官网通知标题不一致")
        matched.append(notice)
    return matched


def _read_mappings(source: Iterable[Mapping[str, Any]] | Path) -> list[Mapping[str, Any]]:
    if isinstance(source, Path):
        mappings: list[Mapping[str, Any]] = []
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetValidationError(f"line {line_number} is not valid JSON") from exc
                if not isinstance(value, dict):
                    raise DatasetValidationError(f"line {line_number} must be a JSON object")
                mappings.append(value)
        return mappings
    return list(source)


def _validate_rows(rows: tuple[AnnotationRow, ...], *, allow_mini_fixture: bool) -> None:
    if not allow_mini_fixture and not 150 <= len(rows) <= 250:
        raise DatasetValidationError("冻结集必须包含 150 至 250 条人工复核记录")
    if not rows:
        raise DatasetValidationError("冻结集不能为空")
    ids: set[str] = set()
    for row in rows:
        if row.id in ids:
            raise DatasetValidationError(f"duplicate annotation id: {row.id}")
        ids.add(row.id)
        if row.reviewed_by != "human":
            raise DatasetValidationError("所有冻结记录必须完成人工复核")
        if not row.original_url.startswith("https://"):
            raise DatasetValidationError("original_url must use HTTPS")
        if not row.duplicate_cluster_id:
            raise DatasetValidationError("duplicate_cluster_id is required")
        if not REQUIRED_PERSONA_IDS.issubset(row.relevance_labels):
            raise DatasetValidationError("缺少四个演示画像的相关性标签")
        if any(label not in {0, 1, 2, 3} for label in row.relevance_labels.values()):
            raise DatasetValidationError("相关性标签必须为 0 至 3")
        if not row.evidence_corrections:
            raise DatasetValidationError("evidence_corrections is required")


def _canonical_bytes(rows: tuple[AnnotationRow, ...]) -> bytes:
    lines = [
        json.dumps(row.to_mapping(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def freeze_annotations(
    source: Iterable[Mapping[str, Any]] | Path,
    output: Path,
    *,
    verified_repository: NoticeRepository | None = None,
) -> FrozenDataset:
    rows = tuple(AnnotationRow.from_mapping(value) for value in _read_mappings(source))
    _validate_rows(rows, allow_mini_fixture=False)
    matched = None
    if verified_repository is not None:
        matched = _matching_verified_notices(rows, verified_repository)
        if matched is None:
            raise DatasetValidationError("冻结记录必须来自已核实官网通知")
    payload = _canonical_bytes(rows)
    digest = hashlib.sha256(payload).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    _manifest_path(output).write_text(f"{digest}\n", encoding="ascii")
    provenance = _provenance_path(output)
    if matched is None:
        provenance.unlink(missing_ok=True)
        return FrozenDataset(rows=rows, sha256=digest, path=output)
    proof_digest = _proof_digest(matched)
    provenance.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corpus_kind": "verified_official_notices",
                "dataset_sha256": digest,
                "verified_row_count": len(matched),
                "proof_digest": proof_digest,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return FrozenDataset(rows=rows, sha256=digest, path=output,
                         corpus_kind="verified_official_notices", proof_digest=proof_digest)


def load_frozen_dataset(
    path: Path,
    *,
    allow_mini_fixture: bool = False,
) -> FrozenDataset:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    manifest = _manifest_path(path)
    if manifest.is_file():
        declared = manifest.read_text(encoding="ascii").strip()
        if declared != digest:
            raise DatasetValidationError("冻结集 SHA-256 与清单不一致")
    elif not allow_mini_fixture:
        raise DatasetValidationError("冻结集缺少 SHA-256 清单")
    rows = tuple(AnnotationRow.from_mapping(value) for value in _read_mappings(path))
    _validate_rows(rows, allow_mini_fixture=allow_mini_fixture)
    if allow_mini_fixture:
        return FrozenDataset(rows=rows, sha256=digest, path=path)
    provenance = _provenance_path(path)
    if not provenance.is_file():
        return FrozenDataset(rows=rows, sha256=digest, path=path)
    try:
        proof = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError("冻结集证明文件无法读取") from exc
    if (
        not isinstance(proof, dict)
        or proof.get("schema_version") != 1
        or proof.get("corpus_kind") != "verified_official_notices"
        or proof.get("dataset_sha256") != digest
        or proof.get("verified_row_count") != len(rows)
        or not isinstance(proof.get("proof_digest"), str)
        or re.fullmatch(r"[0-9a-f]{64}", proof["proof_digest"]) is None
    ):
        raise DatasetValidationError("冻结集证明与数据集不一致或格式无效")
    return FrozenDataset(rows=rows, sha256=digest, path=path,
                         corpus_kind="verified_official_notices",
                         proof_digest=proof["proof_digest"])


def verified_rows_match_repository(
    dataset: FrozenDataset, repository: NoticeRepository
) -> bool:
    if dataset.corpus_kind != "verified_official_notices" or not dataset.proof_digest:
        return False
    try:
        matched = _matching_verified_notices(dataset.rows, repository)
    except DatasetValidationError:
        return False
    return matched is not None and _proof_digest(matched) == dataset.proof_digest


def split_by_cluster(
    dataset: FrozenDataset,
    *,
    train: float,
    validation: float,
    test: float,
    seed: int,
) -> dict[str, list[AnnotationRow]]:
    if not math.isclose(train + validation + test, 1.0):
        raise ValueError("split ratios must sum to 1")
    clusters: dict[str, list[AnnotationRow]] = {}
    for row in dataset.rows:
        clusters.setdefault(row.duplicate_cluster_id, []).append(row)
    cluster_ids = sorted(clusters)
    random.Random(seed).shuffle(cluster_ids)
    train_end = round(len(cluster_ids) * train)
    validation_end = train_end + round(len(cluster_ids) * validation)
    if train > 0 and validation > 0 and test > 0 and len(cluster_ids) >= 3:
        train_end = min(max(train_end, 1), len(cluster_ids) - 2)
        validation_end = min(
            max(validation_end, train_end + 1),
            len(cluster_ids) - 1,
        )
    assignments = {
        "train": cluster_ids[:train_end],
        "validation": cluster_ids[train_end:validation_end],
        "test": cluster_ids[validation_end:],
    }
    return {
        partition: [row for cluster_id in ids for row in clusters[cluster_id]]
        for partition, ids in assignments.items()
    }
