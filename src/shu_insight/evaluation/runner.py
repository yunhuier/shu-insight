from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shu_insight.domain.models import DeadlineConfidence, NoticeRecord, Persona
from shu_insight.evaluation.dataset import (
    REQUIRED_PERSONA_IDS,
    AnnotationRow,
    FrozenDataset,
    split_by_cluster,
)
from shu_insight.evaluation.metrics import (
    accuracy,
    entity_prf,
    important_notice_miss_rate,
    macro_f1,
    ndcg_at_k,
    pairwise_duplicate_prf,
    precision_at_k,
    recall_at_k,
)
from shu_insight.mining.deduplicate import assign_clusters
from shu_insight.mining.fingerprint import simhash
from shu_insight.processing.classify import classify_notice, load_taxonomy
from shu_insight.processing.extract import extract_fields
from shu_insight.processing.normalize import NormalizedDocument
from shu_insight.profiles.service import DEMO_PERSONAS
from shu_insight.ranking.expand import ExpandedQuery, Query, expand_terms
from shu_insight.ranking.relevance import (
    ExactBackend,
    RelevanceBackend,
    SemanticBackend,
    TfidfBackend,
)
from shu_insight.ranking.score import rank_notices
from shu_insight.sources.registry import load_source_registry


@dataclass(frozen=True, slots=True)
class Experiment:
    name: str
    backend: str
    persona: bool
    expansion: bool
    urgency: bool
    dedup: bool


EXPERIMENTS = (
    Experiment("exact_keyword", "exact", False, False, False, False),
    Experiment("tfidf", "tfidf", False, True, False, False),
    Experiment("semantic", "semantic", False, True, False, False),
    Experiment("hybrid_final", "hybrid", True, True, True, True),
    Experiment("ablate_persona", "hybrid", False, True, True, True),
    Experiment("ablate_expansion", "hybrid", True, False, True, True),
    Experiment("ablate_urgency", "hybrid", True, True, False, True),
    Experiment("ablate_dedup", "hybrid", True, True, True, False),
    Experiment("ai_off", "ai", True, True, True, True),
)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    name: str
    metrics: dict[str, float]
    status: str = "ok"
    backend: str = ""
    reason: str = ""
    fitted_ids: tuple[str, ...] = ()
    rankings: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metrics": self.metrics,
            "status": self.status,
            "backend": self.backend,
            "reason": self.reason,
            "fitted_ids": list(self.fitted_ids),
            "rankings": {
                persona_id: list(items) for persona_id, items in self.rankings.items()
            },
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    dataset_sha256: str
    experiments: tuple[ExperimentResult, ...]
    row_count: int
    evidence_kind: str
    formal_metrics_available: bool
    split_ids: dict[str, tuple[str, ...]]
    error_cases: tuple[dict[str, Any], ...]
    corpus_kind: str = "self_test"

    def to_mapping(self) -> dict[str, object]:
        return {
            "dataset_sha256": self.dataset_sha256,
            "row_count": self.row_count,
            "evidence_kind": self.evidence_kind,
            "corpus_kind": self.corpus_kind,
            "formal_metrics_available": self.formal_metrics_available,
            "split_ids": {
                partition: list(row_ids)
                for partition, row_ids in self.split_ids.items()
            },
            "experiments": [result.to_mapping() for result in self.experiments],
            "error_cases": list(self.error_cases),
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_mapping(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _document(row: AnnotationRow) -> str:
    return f"{row.title}\n{row.context_excerpt}"


def _without_expansion(query: ExpandedQuery) -> ExpandedQuery:
    return replace(query, enabled_terms=())


def _formal_evidence_available(dataset: FrozenDataset) -> bool:
    manifest = Path(f"{dataset.path}.sha256")
    return (
        dataset.corpus_kind == "verified_official_notices"
        and dataset.proof_digest is not None
        and 150 <= len(dataset.rows) <= 250
        and len({row.duplicate_cluster_id for row in dataset.rows}) >= 3
        and dataset.path.is_file()
        and manifest.is_file()
        and all(row.reviewed_by == "human" for row in dataset.rows)
    )


def _analyze_rows(
    rows: list[AnnotationRow],
    *,
    taxonomy,
    now: datetime,
) -> tuple[list[NoticeRecord], dict[str, str], dict[str, set[str]]]:
    notices: list[NoticeRecord] = []
    categories: dict[str, str] = {}
    entities: dict[str, set[str]] = {}
    for row in rows:
        text = _document(row)
        document = NormalizedDocument(
            text=text,
            paragraphs=(row.title, row.context_excerpt),
            title=row.title,
        )
        classification = classify_notice(row.title, row.context_excerpt, taxonomy)
        extracted = extract_fields(document, now)
        categories[row.id] = classification.primary
        entity_groups = (
            extracted.education_levels,
            extracted.grades,
            extracted.colleges,
            extracted.majors,
            extracted.locations,
            extracted.materials,
        )
        entities[row.id] = {
            str(item.value) for group in entity_groups for item in group
        }
        notices.append(
            NoticeRecord(
                id=row.id,
                source_id=row.source_id,
                canonical_url=row.original_url,
                title=row.title,
                summary=row.context_excerpt,
                published_at=None,
                deadline_at=extracted.deadline_at,
                deadline_confidence=extracted.deadline_confidence,
                location=(extracted.locations[0].value if extracted.locations else ""),
                education_levels=tuple(item.value for item in extracted.education_levels),
                grades=tuple(item.value for item in extracted.grades),
                colleges=tuple(item.value for item in extracted.colleges),
                majors=tuple(item.value for item in extracted.majors),
                categories=classification.categories or (classification.primary,),
                topic_tags=tuple(sorted(classification.tags)),
                content_fingerprint=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                summary_simhash=simhash(text),
                fetched_at=now,
            )
        )
    return notices, categories, entities


def _predicted_cluster_ids(
    notices: list[NoticeRecord], threshold: float
) -> dict[str, str]:
    assignments = assign_clusters(notices, threshold=threshold)
    return {
        member.id: cluster.id
        for cluster in assignments.clusters
        for member in cluster.members
    }


def _select_dedup_threshold(
    rows: list[AnnotationRow], notices: list[NoticeRecord]
) -> float:
    if len(notices) < 2:
        return 0.99
    truth = [row.duplicate_cluster_id for row in rows]
    best: tuple[float, float, float, float] | None = None
    for threshold in (0.99, 0.97, 0.95, 0.93, 0.90, 0.86, 0.82, 0.78):
        predicted = _predicted_cluster_ids(notices, threshold)
        score = pairwise_duplicate_prf(
            truth,
            [predicted[notice.id] for notice in notices],
        )
        candidate = (score.f1, score.precision, score.recall, threshold)
        if best is None or candidate > best:
            best = candidate
    return best[3] if best is not None else 0.99


def _persona_query(
    persona: Persona,
    *,
    taxonomy,
    expansion: bool,
) -> ExpandedQuery:
    expanded = expand_terms(
        Query(
            tags=tuple((*persona.goals, *persona.interest_tags)),
            keywords=persona.custom_keywords,
        ),
        taxonomy,
    )
    return expanded if expansion else _without_expansion(expanded)


def _ranking_entry(recommendation, label: int) -> dict[str, Any]:
    return {
        "notice_id": recommendation.notice.id,
        "label": label,
        "score": asdict(recommendation.score),
    }


class ExperimentRunner:
    def __init__(
        self,
        dataset: FrozenDataset,
        *,
        now: datetime | None = None,
        semantic_model_path: Path | None = None,
    ) -> None:
        self.dataset = dataset
        self.now = now or datetime.now(UTC)
        if self.now.tzinfo is None:
            self.now = self.now.replace(tzinfo=UTC)
        self.semantic_model_path = semantic_model_path
        self.taxonomy = load_taxonomy(Path("config/taxonomy.toml"))
        self.personas = {persona.id: persona for persona in DEMO_PERSONAS}
        registry = load_source_registry(Path("config/sources.toml"))
        self.authorities = {
            source_id: adapter.config.authority_score
            for source_id, adapter in registry.items()
        }

    def _backend(
        self,
        experiment: Experiment,
        train_documents: list[str],
    ) -> tuple[RelevanceBackend | None, str, str]:
        if experiment.backend == "ai":
            return None, "not_applicable", (
                "外部 AI 不参与确定性排序评估；本实验项仅记录能力边界"
            )
        if experiment.backend == "exact":
            backend: RelevanceBackend = ExactBackend()
            backend.fit(train_documents)
            return backend, "ok", ""
        if experiment.backend == "tfidf":
            backend = TfidfBackend()
            backend.fit(train_documents)
            return backend, "ok", ""

        semantic = SemanticBackend(model_path=self.semantic_model_path)
        capability = semantic.capability()
        if experiment.backend == "semantic":
            if not capability.available:
                return None, "unavailable", capability.reason
            semantic.fit(train_documents)
            return semantic, "ok", ""

        if capability.available:
            semantic.fit(train_documents)
            return semantic, "ok", ""
        fallback = TfidfBackend()
        fallback.fit(train_documents)
        return fallback, "ok", f"semantic_unavailable:{capability.reason};tfidf_fallback"

    def run(self) -> EvaluationReport:
        splits = split_by_cluster(
            self.dataset,
            train=0.70,
            validation=0.15,
            test=0.15,
            seed=42,
        )
        train_rows = list(splits["train"])
        validation_rows = list(splits["validation"])
        test_rows = list(splits["test"])
        if not train_rows or not test_rows:
            raise ValueError(
                "evaluation requires at least one training and one held-out test cluster"
            )
        train_documents = [_document(row) for row in train_rows]
        test_by_id = {row.id: row for row in test_rows}
        gold_clusters = tuple(dict.fromkeys(row.duplicate_cluster_id for row in test_rows))
        important_clusters = {
            row.duplicate_cluster_id for row in test_rows if row.important
        }

        validation_notices, _, _ = _analyze_rows(
            validation_rows,
            taxonomy=self.taxonomy,
            now=self.now,
        )
        test_notices, predicted_categories, predicted_entities = _analyze_rows(
            test_rows,
            taxonomy=self.taxonomy,
            now=self.now,
        )
        dedup_threshold = _select_dedup_threshold(
            validation_rows,
            validation_notices,
        )
        predicted_clusters = _predicted_cluster_ids(test_notices, dedup_threshold)

        truth_categories = [row.category for row in test_rows]
        category_predictions = [predicted_categories[row.id] for row in test_rows]
        entity_score = entity_prf(
            [set(row.gold_entities) for row in test_rows],
            [predicted_entities[row.id] for row in test_rows],
        )
        duplicate_score = pairwise_duplicate_prf(
            [row.duplicate_cluster_id for row in test_rows],
            [predicted_clusters[row.id] for row in test_rows],
        )
        shared_metrics = {
            "classification_accuracy": accuracy(truth_categories, category_predictions),
            "classification_macro_f1": macro_f1(
                truth_categories,
                category_predictions,
            ),
            "entity_precision": entity_score.precision,
            "entity_recall": entity_score.recall,
            "entity_f1": entity_score.f1,
            "duplicate_precision": duplicate_score.precision,
            "duplicate_recall": duplicate_score.recall,
            "duplicate_f1": duplicate_score.f1,
            "source_success_rate": (
                sum(row.source_success for row in test_rows) / len(test_rows)
                if test_rows
                else 0.0
            ),
            "source_failure_rate": (
                sum(not row.source_success for row in test_rows) / len(test_rows)
                if test_rows
                else 0.0
            ),
        }

        results: list[ExperimentResult] = []
        for experiment in EXPERIMENTS:
            backend, status, reason = self._backend(experiment, train_documents)
            if backend is None:
                results.append(
                    ExperimentResult(
                        name=experiment.name,
                        metrics={},
                        status=status,
                        backend=experiment.backend,
                        reason=reason,
                    )
                )
                continue

            rankings: dict[str, tuple[dict[str, Any], ...]] = {}
            precision_scores: list[float] = []
            recall_scores: list[float] = []
            ndcg_scores: list[float] = []
            primary_retrieved: set[str] = set()
            for persona_id in sorted(REQUIRED_PERSONA_IDS):
                persona = self.personas[persona_id]
                query = _persona_query(
                    persona,
                    taxonomy=self.taxonomy,
                    expansion=experiment.expansion,
                )
                scored_notices = (
                    test_notices
                    if experiment.urgency
                    else [
                        replace(
                            notice,
                            deadline_at=None,
                            deadline_confidence=DeadlineConfidence.UNKNOWN,
                        )
                        for notice in test_notices
                    ]
                )
                relevance = {
                    notice.id: backend.score(
                        query,
                        f"{notice.title}\n{notice.summary}",
                    )
                    for notice in scored_notices
                }
                ranked = rank_notices(
                    scored_notices,
                    relevance,
                    self.authorities,
                    self.now,
                    persona=persona if experiment.persona else None,
                )
                if experiment.dedup:
                    unique = []
                    seen: set[str] = set()
                    for recommendation in ranked:
                        cluster_id = predicted_clusters[recommendation.notice.id]
                        if cluster_id in seen:
                            continue
                        seen.add(cluster_id)
                        unique.append(recommendation)
                    ranked = unique

                # Gold clusters are used only to score predictions. They never
                # decide which notices are retrieved, ranked, or merged.
                ideal_by_cluster = {
                    cluster_id: max(
                        row.relevance_labels[persona_id]
                        for row in test_rows
                        if row.duplicate_cluster_id == cluster_id
                    )
                    for cluster_id in gold_clusters
                }
                seen_gold_clusters: set[str] = set()
                labels: list[int] = []
                for recommendation in ranked:
                    cluster_id = test_by_id[recommendation.notice.id].duplicate_cluster_id
                    label = (
                        test_by_id[recommendation.notice.id].relevance_labels[persona_id]
                        if cluster_id not in seen_gold_clusters
                        else 0
                    )
                    labels.append(label)
                    seen_gold_clusters.add(cluster_id)
                binary = [int(label > 0) for label in labels]
                total_relevant = sum(
                    label > 0 for label in ideal_by_cluster.values()
                )
                precision_scores.append(precision_at_k(binary, k=10))
                recall_scores.append(
                    recall_at_k(binary, total_relevant=total_relevant, k=10)
                )
                ndcg_scores.append(
                    ndcg_at_k(
                        list(ideal_by_cluster.values()),
                        labels,
                        k=10,
                    )
                )
                rankings[persona_id] = tuple(
                    _ranking_entry(recommendation, label)
                    for recommendation, label in zip(ranked, labels, strict=True)
                )
                if persona_id == "demo-undergrad-research":
                    primary_retrieved = {
                        recommendation.notice.id for recommendation in ranked[:10]
                    }

            retrieved_clusters = {
                test_by_id[notice_id].duplicate_cluster_id
                for notice_id in primary_retrieved
            }
            metrics = {
                **shared_metrics,
                "precision_at_10": sum(precision_scores) / len(precision_scores),
                "recall_at_10": sum(recall_scores) / len(recall_scores),
                "ndcg_at_10": sum(ndcg_scores) / len(ndcg_scores),
                "important_notice_miss_rate": important_notice_miss_rate(
                    [cluster_id in important_clusters for cluster_id in gold_clusters],
                    [cluster_id in retrieved_clusters for cluster_id in gold_clusters],
                ),
            }
            results.append(
                ExperimentResult(
                    name=experiment.name,
                    metrics=metrics,
                    status=status,
                    backend=backend.name,
                    reason=reason,
                    fitted_ids=(
                        tuple(row.id for row in train_rows)
                        if isinstance(backend, TfidfBackend)
                        else ()
                    ),
                    rankings=rankings,
                )
            )

        errors: list[dict[str, Any]] = []
        for row in test_rows:
            if predicted_categories[row.id] != row.category:
                errors.append(
                    {
                        "notice_id": row.id,
                        "kind": "classification",
                        "expected": row.category,
                        "actual": predicted_categories[row.id],
                    }
                )
            if predicted_entities[row.id] != set(row.gold_entities):
                errors.append(
                    {
                        "notice_id": row.id,
                        "kind": "entity_extraction",
                        "expected": sorted(row.gold_entities),
                        "actual": sorted(predicted_entities[row.id]),
                    }
                )
        hybrid = next(result for result in results if result.name == "hybrid_final")
        retrieved_clusters = {
            test_by_id[item["notice_id"]].duplicate_cluster_id
            for item in hybrid.rankings.get("demo-undergrad-research", ())[:10]
        }
        reported_missed_clusters: set[str] = set()
        for row in test_rows:
            if (
                row.important
                and row.duplicate_cluster_id not in retrieved_clusters
                and row.duplicate_cluster_id not in reported_missed_clusters
            ):
                reported_missed_clusters.add(row.duplicate_cluster_id)
                errors.append(
                    {
                        "notice_id": row.id,
                        "kind": "important_notice_missed",
                        "expected": "retrieved",
                        "actual": "missed",
                    }
                )

        formal = _formal_evidence_available(self.dataset)
        return EvaluationReport(
            dataset_sha256=self.dataset.sha256,
            experiments=tuple(results),
            row_count=len(self.dataset.rows),
            evidence_kind=(
                "formal_frozen_evaluation"
                if formal
                else (
                    "synthetic_self_test"
                    if len(self.dataset.rows) < 150
                    else "unverified_local_dataset"
                )
            ),
            formal_metrics_available=formal,
            corpus_kind=self.dataset.corpus_kind,
            split_ids={
                partition: tuple(row.id for row in rows)
                for partition, rows in splits.items()
            },
            error_cases=tuple(errors[:50]),
        )
