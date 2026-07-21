"""Interchangeable regularizers and controls for the corrected experiments."""

from .distribution import (
    ClassConditionalLogKDEReference,
    ClassConditionalRBFMMD,
    ClassConditionalRBFReferenceMMD,
    HybridQuantumKDEContrastiveReference,
    KDERelationalCoverageReference,
    ModularAblation,
    QuantumCoherenceResidual,
    QuantumDensityMMD,
    QuantumModularContrastiveReference,
    QuantumModularFreeEnergy,
    QuantumModularReference,
    RBFQuantumCoherenceGuidance,
    RelationalKernel,
    TrainableRelationalCoverageReference,
)
from .energy_models import ClassicalPrototypeEnergy, PermutedClassEnergy

__all__ = [
    "ClassConditionalLogKDEReference",
    "ClassConditionalRBFMMD",
    "ClassConditionalRBFReferenceMMD",
    "ClassicalPrototypeEnergy",
    "HybridQuantumKDEContrastiveReference",
    "KDERelationalCoverageReference",
    "ModularAblation",
    "PermutedClassEnergy",
    "QuantumCoherenceResidual",
    "QuantumDensityMMD",
    "QuantumModularFreeEnergy",
    "QuantumModularContrastiveReference",
    "QuantumModularReference",
    "RBFQuantumCoherenceGuidance",
    "RelationalKernel",
    "TrainableRelationalCoverageReference",
]
