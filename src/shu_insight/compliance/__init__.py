"""Compliance gates for public-page collection and AI sanitization."""

from shu_insight.compliance.detector import sanitize_for_ai, scan_sensitive_text
from shu_insight.compliance.policy import evaluate_discovery, evaluate_response

__all__ = [
    "evaluate_discovery",
    "evaluate_response",
    "sanitize_for_ai",
    "scan_sensitive_text",
]

