from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from vqe_gan.config import ExperimentConfig, ExperimentVariant
from vqe_gan.data import create_seeded_data_loader
from vqe_gan.runner import _build_models


def _config(variant: ExperimentVariant) -> ExperimentConfig:
    return ExperimentConfig(
        run_name=variant.value,
        variant=variant,
        regularizer_weight=0.0 if variant is ExperimentVariant.NO_REGULARIZER else 0.1,
    )


def test_all_variants_use_identical_neural_architectures_and_parameter_counts() -> None:
    variants = list(ExperimentVariant)
    structures = []
    counts = []
    for variant in variants:
        uses_quantum = variant in {
            ExperimentVariant.QUANTUM_CONTRASTIVE,
            ExperimentVariant.QUANTUM_PERMUTED,
        }
        generator, discriminator, energy_model = _build_models(
            _config(variant),
            torch.device("cpu"),
            torch.device("cpu") if uses_quantum else None,
        )
        structures.append(
            (tuple(generator.state_dict()), tuple(discriminator.state_dict()))
        )
        counts.append(
            (
                sum(parameter.numel() for parameter in generator.parameters()),
                sum(parameter.numel() for parameter in discriminator.parameters()),
            )
        )
        if energy_model is not None:
            assert sum(parameter.numel() for parameter in energy_model.parameters()) == 0

    assert all(structure == structures[0] for structure in structures[1:])
    assert all(count == counts[0] for count in counts[1:])


def test_seeded_loader_order_is_independent_of_global_rng_consumption() -> None:
    sample_ids = torch.arange(40)
    dataset = TensorDataset(sample_ids)
    first_loader = create_seeded_data_loader(
        dataset,
        batch_size=8,
        seed=53,
        num_workers=0,
        pin_memory=False,
    )
    torch.randn(10_000)
    second_loader = create_seeded_data_loader(
        dataset,
        batch_size=8,
        seed=53,
        num_workers=0,
        pin_memory=False,
    )

    first_order = torch.cat([batch[0] for batch in first_loader])
    second_order = torch.cat([batch[0] for batch in second_loader])
    torch.testing.assert_close(first_order, second_order)
