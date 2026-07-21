from __future__ import annotations

import torch
import torch.nn.functional as functional

from vqe_gan.analysis import classwise_feature_diagnostics, decompose_hybrid_gradients
from vqe_gan.models import SharedQuantumGenerator
from vqe_gan.regularizers import HybridQuantumKDEContrastiveReference


def test_hybrid_gradient_decomposition_is_exact() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.3, -0.2, 0.5]))
    labels = torch.tensor([0, 1], dtype=torch.long)
    kde_logits = torch.stack(
        (
            torch.stack((2 * parameter[0], parameter[1], parameter[2])),
            torch.stack((parameter[0], 3 * parameter[1], -parameter[2])),
        )
    )
    quantum_logits = torch.stack(
        (
            torch.stack((-parameter[1], parameter[2], parameter[0])),
            torch.stack((parameter[2], parameter[0], -parameter[1])),
        )
    )
    decomposition = decompose_hybrid_gradients(
        functional.cross_entropy(kde_logits, labels),
        functional.cross_entropy(0.95 * kde_logits, labels),
        functional.cross_entropy(0.95 * kde_logits + 0.05 * quantum_logits, labels),
        (parameter,),
    )

    torch.testing.assert_close(
        decomposition["total_delta"][0],
        decomposition["scale_effect"][0] + decomposition["quantum_addition"][0],
        atol=1e-7,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        decomposition["identity_residual"][0],
        torch.zeros_like(parameter),
        atol=1e-7,
        rtol=0,
    )
    assert torch.count_nonzero(decomposition["scale_effect"][0]) > 0
    assert torch.count_nonzero(decomposition["quantum_addition"][0]) > 0


def test_classwise_diagnostics_preserve_fid_decomposition() -> None:
    real_features = torch.tensor(
        [[0.0, 0.0], [0.2, 0.1], [1.0, 1.0], [1.2, 0.9]],
    )
    generated_features = torch.tensor(
        [[0.1, 0.0], [0.3, 0.2], [0.9, 1.1], [1.1, 1.0]],
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    logits = torch.tensor(
        [[3.0, 0.0], [2.0, 0.5], [0.0, 3.0], [2.0, 1.0]],
    )

    rows = classwise_feature_diagnostics(
        real_features,
        labels,
        generated_features,
        labels,
        logits,
    )

    assert [row["class"] for row in rows] == [0, 1]
    assert rows[0]["conditional_accuracy"] == 1.0
    assert rows[1]["conditional_accuracy"] == 0.5
    for row in rows:
        assert abs(
            row["class_fid"]
            - row["fid_centroid_component"]
            - row["fid_covariance_component"]
        ) < 1e-7
        assert row["intra_class_diversity"] > 0


def test_contrastive_reference_bypasses_generator_angle_head() -> None:
    torch.manual_seed(17)
    generator = SharedQuantumGenerator()
    generator.eval()
    regularizer = HybridQuantumKDEContrastiveReference(num_classes=3)
    reference = torch.randn(6, 1, 28, 28).tanh()
    reference_labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    regularizer.fit_reference(reference, reference_labels)
    noise = torch.randn(3, generator.latent_dim)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)

    images = generator(noise, labels)
    loss = functional.cross_entropy(regularizer.class_logits(images), labels)
    loss.backward()

    assert generator.input_projection.weight.grad is not None
    assert torch.count_nonzero(generator.input_projection.weight.grad) > 0
    assert all(parameter.grad is None for parameter in generator.angle_head.parameters())
    assert sum(parameter.numel() for parameter in regularizer.parameters()) == 0
