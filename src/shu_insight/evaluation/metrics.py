from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True, slots=True)
class PRF:
    precision: float
    recall: float
    f1: float


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _prf(true_positive: int, false_positive: int, false_negative: int) -> PRF:
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return PRF(precision, recall, f1)


def accuracy(truth: Sequence[Hashable], predicted: Sequence[Hashable]) -> float:
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted lengths differ")
    return _safe_divide(
        sum(left == right for left, right in zip(truth, predicted, strict=True)),
        len(truth),
    )


def macro_f1(truth: Sequence[Hashable], predicted: Sequence[Hashable]) -> float:
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted lengths differ")
    labels = set(truth) | set(predicted)
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            t == label and p == label for t, p in zip(truth, predicted, strict=True)
        )
        false_positive = sum(
            t != label and p == label for t, p in zip(truth, predicted, strict=True)
        )
        false_negative = sum(
            t == label and p != label for t, p in zip(truth, predicted, strict=True)
        )
        scores.append(_prf(true_positive, false_positive, false_negative).f1)
    return _safe_divide(sum(scores), len(scores))


def entity_prf(
    truth: Sequence[set[str]],
    predicted: Sequence[set[str]],
) -> PRF:
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted lengths differ")
    true_positive = sum(
        len(left & right) for left, right in zip(truth, predicted, strict=True)
    )
    false_positive = sum(
        len(right - left) for left, right in zip(truth, predicted, strict=True)
    )
    false_negative = sum(
        len(left - right) for left, right in zip(truth, predicted, strict=True)
    )
    return _prf(true_positive, false_positive, false_negative)


def pairwise_duplicate_prf(
    true_clusters: Sequence[str],
    predicted_clusters: Sequence[str],
) -> PRF:
    if len(true_clusters) != len(predicted_clusters):
        raise ValueError("cluster label lengths differ")
    true_pairs = {
        pair
        for pair in combinations(range(len(true_clusters)), 2)
        if true_clusters[pair[0]] == true_clusters[pair[1]]
    }
    predicted_pairs = {
        pair
        for pair in combinations(range(len(predicted_clusters)), 2)
        if predicted_clusters[pair[0]] == predicted_clusters[pair[1]]
    }
    return _prf(
        len(true_pairs & predicted_pairs),
        len(predicted_pairs - true_pairs),
        len(true_pairs - predicted_pairs),
    )


def precision_at_k(ranked_binary: Sequence[int], *, k: int) -> float:
    selected = ranked_binary[:k]
    return _safe_divide(sum(bool(value) for value in selected), len(selected))


def recall_at_k(ranked_binary: Sequence[int], *, total_relevant: int, k: int) -> float:
    return _safe_divide(sum(bool(value) for value in ranked_binary[:k]), total_relevant)


def _dcg(relevances: Sequence[int], k: int) -> float:
    return sum(
        (2**relevance - 1) / math.log2(index + 2)
        for index, relevance in enumerate(relevances[:k])
    )


def ndcg_at_k(
    ideal_relevances: Sequence[int], ranked_relevances: Sequence[int], *, k: int
) -> float:
    ideal = _dcg(sorted(ideal_relevances, reverse=True), k)
    return _safe_divide(_dcg(ranked_relevances, k), ideal)


def important_notice_miss_rate(
    important: Sequence[bool],
    retrieved: Sequence[bool],
) -> float:
    if len(important) != len(retrieved):
        raise ValueError("important and retrieved lengths differ")
    total = sum(important)
    missed = sum(
        is_important and not was_retrieved
        for is_important, was_retrieved in zip(important, retrieved, strict=True)
    )
    return _safe_divide(missed, total)
