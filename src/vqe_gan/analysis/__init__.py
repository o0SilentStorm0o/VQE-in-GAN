"""Post-hoc experiment analysis utilities."""

from .contrastive import (
    classwise_feature_diagnostics,
    decompose_hybrid_gradients,
    diagnose_contrastive_failure,
)

__all__ = [
    "classwise_feature_diagnostics",
    "decompose_hybrid_gradients",
    "diagnose_contrastive_failure",
]
