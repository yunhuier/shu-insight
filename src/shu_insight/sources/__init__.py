"""Reviewed official-source adapters and registry loading."""

from shu_insight.sources.base import SourceAdapter
from shu_insight.sources.computer import ComputerAdapter
from shu_insight.sources.graduate import GraduateAdapter
from shu_insight.sources.registry import SourceConfigurationError, load_source_registry
from shu_insight.sources.shu_main import ShuMainAdapter
from shu_insight.sources.undergraduate import UndergraduateAdapter

__all__ = [
    "ComputerAdapter",
    "GraduateAdapter",
    "ShuMainAdapter",
    "SourceAdapter",
    "SourceConfigurationError",
    "UndergraduateAdapter",
    "load_source_registry",
]
