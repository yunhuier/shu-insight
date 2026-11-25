from __future__ import annotations

import importlib.util
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from shu_insight.ranking.expand import ExpandedQuery


@dataclass(frozen=True, slots=True)
class RelevanceResult:
    score: float
    matched_terms: frozenset[str]
    backend: str


@dataclass(frozen=True, slots=True)
class BackendCapability:
    available: bool
    reason: str


class RelevanceBackend(Protocol):
    name: str

    def fit(self, documents: Sequence[str]) -> None: ...

    def score(self, query: ExpandedQuery, document: str) -> RelevanceResult: ...


class TextEncoder(Protocol):
    def encode(self, texts: list[str]): ...


def _matching_terms(query: ExpandedQuery, document: str) -> frozenset[str]:
    matches: set[str] = set()
    for term in query.all_terms:
        if term.isascii():
            found = re.search(
                rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
                document,
                re.I,
            )
        else:
            found = re.search(re.escape(term), document)
        if found is not None:
            matches.add(term)
    return frozenset(matches)


def _clamp(score: float) -> float:
    return min(1.0, max(0.0, float(score)))


class ExactBackend:
    name = "exact"

    def fit(self, documents: Sequence[str]) -> None:
        del documents

    def score(self, query: ExpandedQuery, document: str) -> RelevanceResult:
        matched = _matching_terms(query, document)
        denominator = len(query.all_terms)
        score = len(matched) / denominator if denominator else 0.0
        return RelevanceResult(_clamp(score), matched, self.name)


class TfidfBackend:
    name = "tfidf"

    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None

    def fit(self, documents: Sequence[str]) -> None:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), norm="l2")
        vectorizer.fit(documents)
        self.vectorizer = vectorizer

    def score(self, query: ExpandedQuery, document: str) -> RelevanceResult:
        if self.vectorizer is None:
            raise RuntimeError("TF-IDF backend must be fitted before scoring")
        query_text = " ".join(query.all_terms)
        vectors = self.vectorizer.transform([query_text, document])
        score = float(cosine_similarity(vectors[0], vectors[1])[0, 0])
        return RelevanceResult(
            _clamp(score),
            _matching_terms(query, document),
            self.name,
        )


class SemanticBackend:
    name = "semantic"

    def __init__(
        self,
        encoder: TextEncoder | None = None,
        model_path: Path | None = None,
    ) -> None:
        self.encoder = encoder
        self.model_path = model_path

    def capability(self) -> BackendCapability:
        if self.encoder is not None:
            return BackendCapability(True, "injected_encoder")
        if self.model_path is None or not self.model_path.is_dir():
            return BackendCapability(False, "model_not_cached")
        if importlib.util.find_spec("sentence_transformers") is None:
            return BackendCapability(False, "dependency_missing")
        return BackendCapability(True, "model_cached")

    def fit(self, documents: Sequence[str]) -> None:
        del documents

    def _encoder(self) -> TextEncoder:
        if self.encoder is not None:
            return self.encoder
        capability = self.capability()
        if not capability.available:
            raise RuntimeError(capability.reason)
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(
            str(self.model_path),
            local_files_only=True,
        )
        return self.encoder

    def score(self, query: ExpandedQuery, document: str) -> RelevanceResult:
        query_text = " ".join(query.all_terms)
        vectors = np.asarray(self._encoder().encode([query_text, document]), dtype=float)
        if vectors.shape[0] != 2:
            raise ValueError("encoder must return one vector per input text")
        query_vector, document_vector = vectors
        denominator = np.linalg.norm(query_vector) * np.linalg.norm(document_vector)
        similarity = (
            float(np.dot(query_vector, document_vector) / denominator)
            if denominator
            else 0
        )
        return RelevanceResult(
            _clamp(similarity),
            _matching_terms(query, document),
            self.name,
        )
