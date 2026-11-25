"""Deterministic text normalization and knowledge-field extraction."""

from shu_insight.processing.extract import Evidence, ExtractedFields, extract_fields
from shu_insight.processing.normalize import NormalizedDocument, normalize_html

__all__ = [
    "Evidence",
    "ExtractedFields",
    "NormalizedDocument",
    "extract_fields",
    "normalize_html",
]
