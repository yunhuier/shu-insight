from pathlib import Path

import pytest

from shu_insight.processing.classify import load_taxonomy
from shu_insight.ranking.expand import Query, expand_terms


@pytest.fixture
def taxonomy():
    return load_taxonomy(Path("config/taxonomy.toml"))


def test_baoyan_expands_to_reviewable_terms(taxonomy) -> None:
    expanded = expand_terms(Query(tags=("保研",), keywords=()), taxonomy)

    assert {"推免", "推荐免试", "夏令营"}.issubset(expanded.suggested_terms)
    assert expanded.enabled_terms == expanded.suggested_terms
    assert expanded.original_terms == ("保研",)


def test_user_can_disable_a_suggested_term_without_losing_original(taxonomy) -> None:
    expanded = expand_terms(
        Query(tags=("保研",), keywords=("直博",), disabled_terms=("夏令营",)),
        taxonomy,
    )

    assert "夏令营" in expanded.suggested_terms
    assert "夏令营" not in expanded.enabled_terms
    assert expanded.original_terms == ("保研", "直博")


def test_taxonomy_tag_expansion_is_auditable_and_deduplicated(taxonomy) -> None:
    expanded = expand_terms(Query(tags=("科研",), keywords=("科研",)), taxonomy)

    assert expanded.original_terms == ("科研",)
    assert "科研训练" in expanded.enabled_terms
    assert len(expanded.enabled_terms) == len(set(expanded.enabled_terms))
    assert any(evidence.trigger == "科研" for evidence in expanded.evidence)
