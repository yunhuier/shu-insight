"""Duplicate-event mining and immutable notice version comparison."""

from shu_insight.mining.deduplicate import (
    ClusterAssignments,
    DuplicateCluster,
    SimilarityEvidence,
    assign_clusters,
    candidate_pairs,
    duplicate_similarity,
)
from shu_insight.mining.versioning import VersionDiff, compare_versions

__all__ = [
    "ClusterAssignments",
    "DuplicateCluster",
    "SimilarityEvidence",
    "VersionDiff",
    "assign_clusters",
    "candidate_pairs",
    "compare_versions",
    "duplicate_similarity",
]
