from __future__ import annotations

import pytest
import torch

from vqe_gan.quantum.losses import contrastive_energy_loss, raw_target_energy_loss


def test_raw_target_energy_loss_is_batch_mean() -> None:
    energies = torch.tensor([-2.0, 1.0, -0.5])
    torch.testing.assert_close(raw_target_energy_loss(energies), energies.mean())


def test_contrastive_loss_rewards_low_target_energy() -> None:
    labels = torch.tensor([1], dtype=torch.long)
    good = torch.tensor([[1.0, -2.0, 0.5]])
    bad = torch.tensor([[1.0, 2.0, 0.5]])

    assert contrastive_energy_loss(good, labels) < contrastive_energy_loss(bad, labels)


def test_contrastive_loss_rejects_invalid_temperature_and_label_dtype() -> None:
    energies = torch.zeros(2, 3)
    labels = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="temperature"):
        contrastive_energy_loss(energies, labels, temperature=0.0)
    with pytest.raises(TypeError, match="torch.long"):
        contrastive_energy_loss(energies, labels.to(torch.int32))
