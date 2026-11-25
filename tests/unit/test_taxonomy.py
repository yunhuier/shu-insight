from pathlib import Path

from shu_insight.processing.classify import load_taxonomy
from shu_insight.profiles.catalog import majors_for, resolve_major_alias


def test_computer_undergraduate_major_options_are_fixed() -> None:
    assert majors_for("本科生", "计算机工程与科学学院") == (
        "不限专业",
        "大类未分流/暂未确定",
        "计算机科学与技术",
        "智能科学与技术",
        "网络空间安全",
        "人工智能",
    )


def test_other_or_unset_college_has_neutral_major_option() -> None:
    assert majors_for("本科生", "不限学院") == ("不限专业",)
    assert majors_for("不限", "计算机工程与科学学院") == ("不限专业",)


def test_major_aliases_resolve_to_auditable_canonical_names() -> None:
    taxonomy = load_taxonomy(Path("config/taxonomy.toml"))

    assert resolve_major_alias("计科", taxonomy) == "计算机科学与技术"
    assert resolve_major_alias("AI", taxonomy) == "人工智能"
    assert resolve_major_alias("未收录方向", taxonomy) is None


def test_taxonomy_has_exact_primary_category_set() -> None:
    taxonomy = load_taxonomy(Path("config/taxonomy.toml"))

    assert set(taxonomy.categories) == {
        "教学教务",
        "科研学术",
        "竞赛创新",
        "奖助评优",
        "就业实习",
        "升学推免",
        "校园事务",
        "讲座活动",
    }
