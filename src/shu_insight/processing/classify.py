from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Taxonomy:
    categories: dict[str, tuple[str, ...]]
    tags: dict[str, tuple[str, ...]]
    majors: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class MatchedTerm:
    label: str
    term: str
    field: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class CategoryScore:
    category: str
    score: float
    matched_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Classification:
    primary: str
    categories: tuple[str, ...]
    tags: frozenset[str]
    major_mentions: tuple[str, ...]
    scores: tuple[CategoryScore, ...]
    matches: tuple[MatchedTerm, ...]


def _load_keyword_tables(
    document: dict[str, Any], section: str
) -> dict[str, tuple[str, ...]]:
    raw_table = document.get(section)
    if not isinstance(raw_table, dict) or not raw_table:
        raise ValueError(f"taxonomy section {section!r} must be a non-empty table")
    result: dict[str, tuple[str, ...]] = {}
    for label, raw_entry in raw_table.items():
        if not isinstance(raw_entry, dict):
            raise ValueError(f"taxonomy entry {section}.{label} must be a table")
        raw_keywords = raw_entry.get("keywords" if section != "majors" else "aliases")
        if not isinstance(raw_keywords, list) or not all(
            isinstance(keyword, str) and keyword.strip() for keyword in raw_keywords
        ):
            raise ValueError(f"taxonomy entry {section}.{label} has invalid keywords")
        result[label] = tuple(keyword.strip() for keyword in raw_keywords)
    return result


def load_taxonomy(path: Path) -> Taxonomy:
    with path.open("rb") as taxonomy_file:
        document = tomllib.load(taxonomy_file)
    return Taxonomy(
        categories=_load_keyword_tables(document, "categories"),
        tags=_load_keyword_tables(document, "tags"),
        majors=_load_keyword_tables(document, "majors"),
    )


def _find_term(text: str, term: str) -> re.Match[str] | None:
    if term.isascii():
        return re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text, re.I)
    return re.search(re.escape(term), text)


def _matches_for_label(
    label: str,
    terms: tuple[str, ...],
    title: str,
    text: str,
) -> tuple[MatchedTerm, ...]:
    results: list[MatchedTerm] = []
    for field_name, field_text in (("title", title), ("text", text)):
        occupied: list[tuple[int, int]] = []
        for term in sorted(terms, key=len, reverse=True):
            match = _find_term(field_text, term)
            if match is None or any(
                match.start() < end and match.end() > start for start, end in occupied
            ):
                continue
            occupied.append((match.start(), match.end()))
            results.append(
                MatchedTerm(
                    label=label,
                    term=field_text[match.start() : match.end()],
                    field=field_name,
                    start=match.start(),
                    end=match.end(),
                )
            )
    return tuple(results)


def classify_notice(title: str, text: str, taxonomy: Taxonomy) -> Classification:
    category_scores: list[CategoryScore] = []
    all_matches: list[MatchedTerm] = []
    for category, terms in taxonomy.categories.items():
        matches = _matches_for_label(category, terms, title, text)
        if not matches:
            continue
        score = sum(2.0 if match.field == "title" else 1.0 for match in matches)
        category_scores.append(
            CategoryScore(
                category=category,
                score=score,
                matched_terms=tuple(dict.fromkeys(match.term for match in matches)),
            )
        )
        all_matches.extend(matches)
    category_scores.sort(key=lambda item: -item.score)

    matched_tags: set[str] = set()
    for tag, terms in taxonomy.tags.items():
        matches = _matches_for_label(tag, terms, title, text)
        if matches:
            matched_tags.add(tag)
            all_matches.extend(matches)

    major_mentions: list[str] = []
    for major, aliases in taxonomy.majors.items():
        matches = _matches_for_label(major, (major, *aliases), title, text)
        if matches:
            major_mentions.append(major)
            all_matches.extend(matches)

    primary = category_scores[0].category if category_scores else "校园事务"
    return Classification(
        primary=primary,
        categories=tuple(score.category for score in category_scores),
        tags=frozenset(matched_tags),
        major_mentions=tuple(major_mentions),
        scores=tuple(category_scores),
        matches=tuple(all_matches),
    )
