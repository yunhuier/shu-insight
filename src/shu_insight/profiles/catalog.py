from __future__ import annotations

from shu_insight.processing.classify import Taxonomy

_COMPUTER_COLLEGE = "计算机工程与科学学院"
_MAJOR_OPTIONS: dict[tuple[str, str], tuple[str, ...]] = {
    ("本科生", _COMPUTER_COLLEGE): (
        "不限专业",
        "大类未分流/暂未确定",
        "计算机科学与技术",
        "智能科学与技术",
        "网络空间安全",
        "人工智能",
    ),
    ("硕士生", _COMPUTER_COLLEGE): (
        "不限专业",
        "计算机科学与技术",
        "软件工程",
        "电子信息",
    ),
    ("博士生", _COMPUTER_COLLEGE): (
        "不限专业",
        "计算机科学与技术",
    ),
}


def majors_for(level: str, college: str) -> tuple[str, ...]:
    return _MAJOR_OPTIONS.get((level, college), ("不限专业",))


def resolve_major_alias(value: str, taxonomy: Taxonomy) -> str | None:
    normalized = value.strip().casefold()
    for canonical, aliases in taxonomy.majors.items():
        candidates = (canonical, *aliases)
        if any(normalized == candidate.casefold() for candidate in candidates):
            return canonical
    return None
