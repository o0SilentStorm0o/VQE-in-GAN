"""Interchangeable regularizers and controls for the corrected experiments."""

from .distribution import (
    ClassConditionalLogKDEReference,
    ClassConditionalRBFMMD,
    ClassConditionalRBFReferenceMMD,
    HybridQuantumKDEContrastiveReference,
    ModularAblation,
    QuantumCoherenceResidual,
    QuantumDensityMMD,
    QuantumModularContrastiveReference,
    QuantumModularFreeEnergy,
    QuantumModularReference,
    RBFQuantumCoherenceGuidance,
)
from .energy_models import ClassicalPrototypeEnergy, PermutedClassEnergy

__all__ = [
    "ClassConditionalLogKDEReference",
    "ClassConditionalRBFMMD",
    "ClassConditionalRBFReferenceMMD",
    "ClassicalPrototypeEnergy",
    "HybridQuantumKDEContrastiveReference",
    "ModularAblation",
    "PermutedClassEnergy",
    "QuantumCoherenceResidual",
    "QuantumDensityMMD",
    "QuantumModularFreeEnergy",
    "QuantumModularContrastiveReference",
    "QuantumModularReference",
    "RBFQuantumCoherenceGuidance",
]
