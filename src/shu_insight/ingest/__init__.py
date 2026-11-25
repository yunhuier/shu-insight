"""Polite, isolated ingestion primitives."""

from shu_insight.ingest.http_client import PoliteHttpClient, SourceCircuitBreaker
from shu_insight.ingest.pipeline import IngestionPipeline, UpdateEvent, UpdateReport

__all__ = [
    "IngestionPipeline",
    "PoliteHttpClient",
    "SourceCircuitBreaker",
    "UpdateEvent",
    "UpdateReport",
]
