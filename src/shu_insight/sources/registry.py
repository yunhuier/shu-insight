from __future__ import annotations

import re
import tomllib
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import soupsieve

from shu_insight.domain.models import SourceConfig
from shu_insight.sources.base import SourceAdapter
from shu_insight.sources.computer import ComputerAdapter
from shu_insight.sources.graduate import GraduateAdapter
from shu_insight.sources.shu_main import ShuMainAdapter
from shu_insight.sources.undergraduate import UndergraduateAdapter

_ADAPTER_TYPES: dict[str, type[SourceAdapter]] = {
    "shu_main": ShuMainAdapter,
    "undergraduate": UndergraduateAdapter,
    "graduate": GraduateAdapter,
    "computer": ComputerAdapter,
}


class SourceConfigurationError(ValueError):
    """Raised when a source expands beyond its reviewed access scope."""


def _required_text(raw: dict[str, Any], field: str, source_id: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SourceConfigurationError(f"source {source_id!r} requires non-empty {field}")
    return value.strip()


def _parse_source(raw: dict[str, Any]) -> SourceConfig:
    source_id = _required_text(raw, "id", "<unknown>")
    name = _required_text(raw, "name", source_id)
    base_url = _required_text(raw, "base_url", source_id)
    list_url = _required_text(raw, "list_url", source_id)

    base = urlsplit(base_url)
    listing = urlsplit(list_url)
    if base.scheme.lower() != "https" or listing.scheme.lower() != "https":
        raise SourceConfigurationError(f"source {source_id!r} URLs must use HTTPS")
    hosts_differ = base.hostname and listing.hostname and (
        base.hostname.lower() != listing.hostname.lower()
    )
    if not base.hostname or not listing.hostname or hosts_differ:
        raise SourceConfigurationError(f"source {source_id!r} URLs must use the same host")

    raw_prefixes = raw.get("allowed_path_prefixes")
    if not isinstance(raw_prefixes, list) or not raw_prefixes:
        raise SourceConfigurationError(f"source {source_id!r} requires an allowed path list")
    if not all(isinstance(prefix, str) and prefix.startswith("/") for prefix in raw_prefixes):
        raise SourceConfigurationError(f"source {source_id!r} has an invalid allowed path")
    prefixes = tuple(raw_prefixes)
    if not any(listing.path.startswith(prefix) for prefix in prefixes):
        raise SourceConfigurationError(f"source {source_id!r} list URL is outside its allowed path")

    try:
        authority_score = float(raw["authority_score"])
        minimum_interval = float(raw["minimum_interval_seconds"])
        reviewed_on = date.fromisoformat(_required_text(raw, "reviewed_on", source_id))
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceConfigurationError(
            f"source {source_id!r} has invalid numeric/date fields"
        ) from exc
    if not 0.0 <= authority_score <= 1.0:
        raise SourceConfigurationError(
            f"source {source_id!r} authority score must be between 0 and 1"
        )
    if minimum_interval < 2.0:
        raise SourceConfigurationError(f"source {source_id!r} interval must be at least 2 seconds")

    pagination_selector = None
    pagination_url_pattern = None
    pagination_reviewed_on = None
    pagination_link_text = None
    pagination_fields = (
        "pagination_selector", "pagination_url_pattern", "pagination_reviewed_on",
        "pagination_link_text",
    )
    if any(field in raw for field in pagination_fields):
        try:
            pagination_selector = _required_text(raw, "pagination_selector", source_id)
            pagination_url_pattern = _required_text(raw, "pagination_url_pattern", source_id)
            pagination_reviewed_on = date.fromisoformat(
                _required_text(raw, "pagination_reviewed_on", source_id)
            )
            if "pagination_link_text" in raw:
                pagination_link_text = _required_text(raw, "pagination_link_text", source_id)
            soupsieve.compile(pagination_selector)
            pattern = re.compile(pagination_url_pattern)
            if not pattern.fullmatch(listing.path):
                raise ValueError("pattern does not include the reviewed first list path")
        except (ValueError, TypeError, re.error, soupsieve.SelectorSyntaxError) as exc:
            raise SourceConfigurationError(
                f"source {source_id!r} has invalid pagination contract"
            ) from exc

    return SourceConfig(
        id=source_id,
        name=name,
        base_url=base_url,
        list_url=list_url,
        allowed_path_prefixes=prefixes,
        authority_score=authority_score,
        minimum_interval_seconds=minimum_interval,
        reviewed_on=reviewed_on,
        pagination_selector=pagination_selector,
        pagination_url_pattern=pagination_url_pattern,
        pagination_reviewed_on=pagination_reviewed_on,
        pagination_link_text=pagination_link_text,
    )


def load_source_registry(path: Path) -> dict[str, SourceAdapter]:
    """Load and validate the complete source allowlist before any network request."""

    with path.open("rb") as config_file:
        document = tomllib.load(config_file)
    raw_sources = document.get("source")
    if not isinstance(raw_sources, list):
        raise SourceConfigurationError("configuration must contain one or more [[source]] entries")

    registry: dict[str, SourceAdapter] = {}
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise SourceConfigurationError("each source entry must be a table")
        config = _parse_source(raw)
        if config.id in registry:
            raise SourceConfigurationError(f"duplicate source id: {config.id}")
        adapter_type = _ADAPTER_TYPES.get(config.id, SourceAdapter)
        registry[config.id] = adapter_type(config)
    return registry
