"""Classical controls and transformations for all-class energy matrices."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor, nn


class AllClassEnergyModel(Protocol):
    def all_energies(self, angles: Tensor) -> Tensor: ...


class ClassicalPrototypeEnergy(nn.Module):
    """Parameter-free periodic distance to deterministic class prototypes.

    Every class has one distinct minimum and the same per-class landscape. The model consumes the
    same circuit-angle vector as the quantum energy model but contains no trainable parameters.
    """

    def __init__(self, *, num_classes: int = 10, num_angles: int = 16) -> None:
        super().__init__()
        if num_classes < 2 or num_angles < 1:
            raise ValueError("num_classes must be at least 2 and num_angles must be positive")
        class_indices = torch.arange(num_classes, dtype=torch.float64).unsqueeze(1)
        angle_offsets = (3 * torch.arange(num_angles, dtype=torch.float64)).unsqueeze(0)
        phase_indices = torch.remainder(class_indices + angle_offsets, num_classes)
        prototypes = (2 * phase_indices / num_classes - 1) * torch.pi
        self.register_buffer("prototypes", prototypes, persistent=True)

    @property
    def num_classes(self) -> int:
        return self.prototypes.shape[0]

    @property
    def num_angles(self) -> int:
        return self.prototypes.shape[1]

    def all_energies(self, angles: Tensor) -> Tensor:
        if angles.ndim != 2 or angles.shape[1] != self.num_angles:
            raise ValueError(
                f"angles must have shape (batch, {self.num_angles}), got {tuple(angles.shape)}"
            )
        prototypes = self.prototypes.to(device=angles.device, dtype=angles.dtype)
        differences = angles.unsqueeze(1) - prototypes.unsqueeze(0)
        return (1 - torch.cos(differences)).mean(dim=-1)


class PermutedClassEnergy(nn.Module):
    """Apply a fixed cyclic permutation to the class columns of another energy model."""

    def __init__(self, energy_model: AllClassEnergyModel, *, num_classes: int = 10) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if isinstance(energy_model, nn.Module):
            self.energy_model = energy_model
        else:
            raise TypeError("energy_model must be a torch module")
        permutation = torch.roll(torch.arange(num_classes), shifts=1)
        self.register_buffer("permutation", permutation, persistent=True)

    def all_energies(self, angles: Tensor) -> Tensor:
        energies = self.energy_model.all_energies(angles)
        if energies.ndim != 2 or energies.shape[1] != self.permutation.numel():
            raise ValueError("wrapped model returned an incompatible class dimension")
        permutation = self.permutation.to(energies.device)
        return energies.index_select(1, permutation)
