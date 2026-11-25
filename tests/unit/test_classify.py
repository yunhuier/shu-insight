from pathlib import Path

import pytest

from shu_insight.processing.classify import classify_notice, load_taxonomy


@pytest.fixture
def taxonomy():
    return load_taxonomy(Path("config/taxonomy.toml"))


def test_classification_keeps_competition_and_research_tags(taxonomy) -> None:
    result = classify_notice(
        "大学生创新竞赛报名通知",
        "优秀项目可进入科研训练计划",
        taxonomy,
    )

    assert result.primary == "竞赛创新"
    assert {"竞赛", "科研"}.issubset(result.tags)
    assert {score.category for score in result.scores} >= {"竞赛创新", "科研学术"}
    assert any(match.term == "竞赛" and match.field == "title" for match in result.matches)


def test_classifier_normalizes_major_alias_but_does_not_force_category(taxonomy) -> None:
    result = classify_notice("实验室开放通知", "欢迎计科和网安同学参加", taxonomy)

    assert set(result.major_mentions) == {"计算机科学与技术", "网络空间安全"}


def test_no_major_mention_remains_neutral(taxonomy) -> None:
    result = classify_notice("图书馆开放时间调整", "请按现场安排有序入馆", taxonomy)

    assert result.major_mentions == ()
    assert all(match.label not in taxonomy.majors for match in result.matches)
