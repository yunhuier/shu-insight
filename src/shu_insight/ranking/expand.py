from __future__ import annotations

from dataclasses import dataclass

from shu_insight.processing.classify import Taxonomy

_CURATED_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "保研": ("推免", "推荐免试", "夏令营", "预推免"),
    "推免": ("保研", "推荐免试", "夏令营", "预推免"),
    "科研": ("科研训练", "学术", "论文", "实验室", "研究项目"),
    "竞赛": ("大赛", "比赛", "创新赛", "挑战杯"),
    "就业": ("招聘", "实习", "宣讲会", "双选会", "岗位"),
}


@dataclass(frozen=True, slots=True)
class Query:
    tags: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    disabled_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExpansionEvidence:
    trigger: str
    suggested_term: str
    source: str


@dataclass(frozen=True, slots=True)
class ExpandedQuery:
    original_terms: tuple[str, ...]
    suggested_terms: tuple[str, ...]
    enabled_terms: tuple[str, ...]
    evidence: tuple[ExpansionEvidence, ...]

    @property
    def all_terms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.original_terms, *self.enabled_terms)))


def _normalized_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    unique: dict[str, str] = {}
    for value in values:
        stripped = value.strip()
        if stripped:
            unique.setdefault(stripped.casefold(), stripped)
    return tuple(unique.values())


def expand_terms(query: Query, taxonomy: Taxonomy) -> ExpandedQuery:
    original = _normalized_unique((*query.tags, *query.keywords))
    original_folded = {term.casefold() for term in original}
    suggestions: dict[str, str] = {}
    evidence: list[ExpansionEvidence] = []

    for trigger in original:
        candidates: list[tuple[str, str]] = [
            (term, "curated") for term in _CURATED_EXPANSIONS.get(trigger, ())
        ]
        candidates.extend((term, "taxonomy_tag") for term in taxonomy.tags.get(trigger, ()))
        candidates.extend(
            (term, "taxonomy_category") for term in taxonomy.categories.get(trigger, ())
        )
        for term, source in candidates:
            key = term.casefold()
            if key in original_folded or key in suggestions:
                continue
            suggestions[key] = term
            evidence.append(ExpansionEvidence(trigger, term, source))

    disabled = {term.strip().casefold() for term in query.disabled_terms}
    suggested_terms = tuple(suggestions.values())
    enabled_terms = tuple(
        term for term in suggested_terms if term.casefold() not in disabled
    )
    return ExpandedQuery(
        original_terms=original,
        suggested_terms=suggested_terms,
        enabled_terms=enabled_terms,
        evidence=tuple(evidence),
    )
