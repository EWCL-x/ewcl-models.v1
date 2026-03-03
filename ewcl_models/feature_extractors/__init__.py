"""Feature extractors for EWCL publication models."""

from ewcl_models.feature_extractors.sequence_features import (
    build_ewclv1_features,
    build_ewclv1_features as build_sequence_features,  # backwards-compat alias
    FeatureBlock,
)
from ewcl_models.feature_extractors.structure_features import (
    compute_structure_features,
)

__all__ = [
    "build_ewclv1_features",
    "build_sequence_features",  # backwards-compat alias
    "FeatureBlock",
    "compute_structure_features",
]
