"""Loss functions for class-conditioned quantum energies."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor


def raw_target_energy_loss(target_energies: Tensor) -> Tensor:
    """Historical objective: minimize the selected class energy directly."""

    if target_energies.ndim != 1:
        raise ValueError("target_energies must have shape (batch,)")
    return target_energies.mean()


def contrastive_energy_loss(
    all_energies: Tensor,
    class_labels: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Prefer the target Hamiltonian over every non-target Hamiltonian.

    Lower energy is better, so negative energies become the classification logits.
    """

    if all_energies.ndim != 2:
        raise ValueError("all_energies must have shape (batch, classes)")
    if class_labels.ndim != 1 or class_labels.shape[0] != all_energies.shape[0]:
        raise ValueError("class_labels must have shape (batch,)")
    if class_labels.dtype != torch.long:
        raise TypeError("class_labels must have dtype torch.long")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return functional.cross_entropy(-all_energies / temperature, class_labels)
