from __future__ import annotations

from pathlib import Path

import pytest

from shu_insight.processing.classify import load_taxonomy
from shu_insight.ranking.expand import ExpandedQuery, Query, expand_terms
from shu_insight.ranking.relevance import ExactBackend, SemanticBackend, TfidfBackend


@pytest.fixture
def expanded_query():
    taxonomy = load_taxonomy(Path("config/taxonomy.toml"))
    return expand_terms(Query(tags=("保研",), keywords=("直博",)), taxonomy)


def test_exact_backend_returns_matched_terms_and_bounded_score(expanded_query) -> None:
    result = ExactBackend().score(
        expanded_query,
        "本次推荐免试夏令营面向有直博意愿的本科生开放报名。",
    )

    assert result.backend == "exact"
    assert {"推荐免试", "夏令营", "直博"}.issubset(result.matched_terms)
    assert 0.0 <= result.score <= 1.0


def test_exact_title_and_summary_evidence_stays_separate() -> None:
    query = ExpandedQuery(original_terms=("AI", "科研"), suggested_terms=(),
                          enabled_terms=(), evidence=())
    backend = ExactBackend()
    title = backend.score(query, "RAIL课程通知")
    summary = backend.score(query, "科研项目申请")

    assert title.matched_terms == frozenset()
    assert summary.matched_terms == frozenset({"科研"})


def test_tfidf_backend_is_deterministic_and_ranks_relevant_text_higher(
    expanded_query,
) -> None:
    relevant = "推荐免试夏令营与直博项目报名安排"
    unrelated = "图书馆开放时间调整与校园卡服务说明"
    backend = TfidfBackend()
    backend.fit([relevant, unrelated])

    first = backend.score(expanded_query, relevant)
    second = backend.score(expanded_query, relevant)
    other = backend.score(expanded_query, unrelated)

    assert first == second
    assert first.backend == "tfidf"
    assert 0.0 <= other.score < first.score <= 1.0


def test_semantic_backend_reports_unavailable_without_downloading(tmp_path: Path) -> None:
    backend = SemanticBackend(model_path=tmp_path / "missing-model")

    assert backend.capability().available is False
    assert backend.capability().reason == "model_not_cached"


class KeywordEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if ("推免" in text or "保研" in text) else [0.0, 1.0]
            for text in texts
        ]


def test_semantic_backend_uses_injected_encoder_with_stable_vectors(
    expanded_query,
) -> None:
    backend = SemanticBackend(encoder=KeywordEncoder())

    matching = backend.score(expanded_query, "推免政策说明")
    unrelated = backend.score(expanded_query, "校园体育活动")

    assert backend.capability().available is True
    assert matching.backend == "semantic"
    assert matching.score == pytest.approx(1.0)
    assert unrelated.score == pytest.approx(0.0)
    assert matching == backend.score(expanded_query, "推免政策说明")
