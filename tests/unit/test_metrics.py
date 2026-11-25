import pytest

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


def test_classification_metrics_known_example() -> None:
    truth = ["竞赛", "科研", "竞赛", "就业"]
    predicted = ["竞赛", "科研", "就业", "就业"]

    assert accuracy(truth, predicted) == pytest.approx(0.75)
    assert macro_f1(truth, predicted) == pytest.approx(7 / 9)


def test_entity_metrics_use_micro_counts() -> None:
    result = entity_prf(
        [{"本科生", "宝山校区"}, {"研究生"}],
        [{"本科生"}, {"研究生", "延长校区"}],
    )

    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 / 3)


def test_pairwise_duplicate_metrics_compare_cluster_pairs() -> None:
    result = pairwise_duplicate_prf(
        ["a", "a", "b", "c"],
        ["x", "x", "x", "z"],
    )

    assert result.precision == pytest.approx(1 / 3)
    assert result.recall == pytest.approx(1.0)


def test_ranking_metrics_known_examples() -> None:
    ranked_binary = [1, 0, 1, 0]

    assert precision_at_k(ranked_binary, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(ranked_binary, total_relevant=2, k=3) == pytest.approx(1.0)
    assert ndcg_at_k([3, 2, 0], [3, 0, 2], k=10) == pytest.approx(0.95583, rel=1e-4)


def test_important_notice_miss_rate() -> None:
    assert important_notice_miss_rate([True, True, False], [True, False, False]) == 0.5
