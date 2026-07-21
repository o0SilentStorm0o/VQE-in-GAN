from __future__ import annotations

import torch

from vqe_gan.regularizers import (
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


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(101)
    real = torch.randn(6, 1, 28, 28).tanh()
    generated = (real + 0.2 * torch.randn_like(real)).clamp(-1, 1).requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    return generated, real, labels


def test_modular_free_energy_is_zero_only_for_matching_class_states() -> None:
    generated, real, labels = _batch()
    regularizer = QuantumModularFreeEnergy()

    matching = regularizer(real.clone().requires_grad_(True), real, labels)
    different = regularizer(generated, real, labels)

    assert abs(matching.item()) < 2e-5
    assert different.item() > matching.item() + 1e-5
    different.backward()
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()
    assert torch.count_nonzero(generated.grad) > 0


def test_dephasing_removes_final_rz_phase_information() -> None:
    real = torch.zeros(2, 1, 4, 4)
    generated = real.clone()
    real[:, :, 0, :] = 0.5
    generated[:, :, 0, :] = 0.5
    real[:, :, 3, :] = -0.5
    generated[:, :, 3, :] = 0.5
    labels = torch.zeros(2, dtype=torch.long)
    full = QuantumModularFreeEnergy(ablation=ModularAblation.FULL)
    dephased = QuantumModularFreeEnergy(ablation=ModularAblation.DEPHASED)

    full_loss = full(generated, real, labels)
    dephased_loss = dephased(generated, real, labels)

    assert full_loss > 1e-4
    assert abs(dephased_loss.item()) < 1e-6


def test_distribution_controls_are_zero_on_identical_batches_and_parameter_free() -> None:
    _, real, labels = _batch()
    regularizers = [
        QuantumDensityMMD(),
        ClassConditionalRBFMMD(),
    ]

    for regularizer in regularizers:
        loss = regularizer(real, real, labels)
        assert abs(loss.item()) < 1e-6
        assert sum(parameter.numel() for parameter in regularizer.parameters()) == 0


def test_modular_ablations_keep_the_same_parameter_free_circuit() -> None:
    for ablation in ModularAblation:
        regularizer = QuantumModularFreeEnergy(ablation=ablation)
        assert sum(parameter.numel() for parameter in regularizer.parameters()) == 0
        assert regularizer.backend.num_parameters == 16


def test_coherence_residual_is_nonnegative_and_vanishes_for_a_match() -> None:
    generated, real, labels = _batch()
    residual = QuantumCoherenceResidual()

    matching = residual(real, real, labels)
    different = residual(generated, real, labels)

    assert abs(matching.item()) < 2e-5
    assert different.item() >= -2e-5
    different.backward()
    assert generated.grad is not None
    assert torch.count_nonzero(generated.grad) > 0


def test_coherence_guidance_exposes_classical_and_quantum_losses_separately() -> None:
    generated, real, labels = _batch()
    guidance = RBFQuantumCoherenceGuidance()

    classical, quantum = guidance.loss_components(generated, real, labels)

    torch.testing.assert_close(guidance(generated, real, labels), classical + quantum)
    assert classical > 0
    assert quantum >= -2e-5
    assert sum(parameter.numel() for parameter in guidance.parameters()) == 0


def test_fitted_reference_regularizers_match_their_balanced_reference_bank() -> None:
    _, real, labels = _batch()
    regularizers = [
        QuantumModularReference(num_classes=3),
        ClassConditionalRBFReferenceMMD(num_classes=3),
    ]

    for regularizer in regularizers:
        regularizer.fit_reference(real, labels)
        assert regularizer.reference_is_fitted
        loss = regularizer(real.clone().requires_grad_(True), real, labels)
        assert abs(loss.item()) < 2e-5
        assert sum(parameter.numel() for parameter in regularizer.parameters()) == 0


def test_contrastive_modular_observables_are_centered_normalized_and_differentiable() -> None:
    generated, real, labels = _batch()
    regularizer = QuantumModularContrastiveReference(num_classes=3)
    regularizer.fit_reference(real, labels)

    observables = regularizer.reference_observables
    traces = torch.diagonal(observables, dim1=-2, dim2=-1).sum(-1)
    norms = observables.abs().square().sum(dim=(-2, -1)).sqrt()
    loss = regularizer(generated, real, labels)

    torch.testing.assert_close(traces, torch.zeros_like(traces), atol=2e-5, rtol=0)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=2e-5, rtol=0)
    assert loss.isfinite()
    loss.backward()
    assert generated.grad is not None
    assert torch.count_nonzero(generated.grad) > 0
    assert sum(parameter.numel() for parameter in regularizer.parameters()) == 0


def test_contrastive_controls_share_reference_input_and_expose_calibrated_logits() -> None:
    generated, real, labels = _batch()
    classical = ClassConditionalLogKDEReference(num_classes=3)
    hybrid = HybridQuantumKDEContrastiveReference(
        num_classes=3,
        quantum_mixture_weight=0.2,
    )
    classical.fit_reference(real, labels)
    hybrid.fit_reference(real, labels)

    classical_logits = classical.class_logits(generated)
    quantum_logits = hybrid.quantum.class_logits(generated)
    hybrid_logits = hybrid.class_logits(generated)

    assert classical_logits.shape == (generated.shape[0], 3)
    torch.testing.assert_close(
        hybrid_logits,
        0.2 * quantum_logits + 0.8 * hybrid.classical.class_logits(generated),
    )
    assert classical(generated, real, labels).isfinite()
    assert hybrid(generated, real, labels).isfinite()
    assert sum(parameter.numel() for parameter in classical.parameters()) == 0
    assert sum(parameter.numel() for parameter in hybrid.parameters()) == 0


def test_contrastive_quantum_ablations_preserve_the_fixed_pixel_assignment() -> None:
    expected = torch.tensor([5, 7, 15, 13, 0, 2, 10, 8, 1, 3, 11, 9, 4, 6, 14, 12])
    for ablation in (ModularAblation.FULL, ModularAblation.PRODUCT, ModularAblation.DEPHASED):
        regularizer = QuantumModularContrastiveReference(
            num_classes=3,
            ablation=ablation,
        )
        torch.testing.assert_close(regularizer.pixel_permutation.cpu(), expected)


def test_trainable_relational_coverage_reaches_images_and_angle_residuals() -> None:
    generated, real, labels = _batch()
    angle_residuals = torch.zeros(6, 16, requires_grad=True)
    regularizer = TrainableRelationalCoverageReference(num_classes=3)
    regularizer.fit_reference(real, labels)

    loss = regularizer(generated, angle_residuals, labels)
    loss.backward()

    assert loss.isfinite()
    assert generated.grad is not None
    assert torch.count_nonzero(generated.grad) > 0
    assert angle_residuals.grad is not None
    assert torch.count_nonzero(angle_residuals.grad) > 0
    assert sum(parameter.numel() for parameter in regularizer.parameters()) == 0


def test_zero_relational_residual_reproduces_fixed_image_angles() -> None:
    generated, real, labels = _batch()
    del real, labels
    regularizer = TrainableRelationalCoverageReference(num_classes=3)
    residuals = torch.zeros(generated.shape[0], 16)

    torch.testing.assert_close(
        regularizer.generated_angles(generated, residuals),
        regularizer.base_angles(generated),
        atol=0,
        rtol=0,
    )


def test_matched_classical_coverage_uses_fitted_periodic_bandwidth() -> None:
    generated, real, labels = _batch()
    residuals = torch.zeros(6, 16, requires_grad=True)
    regularizer = TrainableRelationalCoverageReference(
        num_classes=3,
        kernel=RelationalKernel.CLASSICAL_PERIODIC_RBF,
    )
    regularizer.fit_reference(real, labels)

    loss = regularizer(generated, residuals, labels)
    loss.backward()

    assert regularizer.classical_sigma_squared.isfinite()
    assert regularizer.classical_sigma_squared > 0
    assert generated.grad is not None
    assert torch.count_nonzero(generated.grad) > 0
    assert residuals.grad is not None
    assert torch.count_nonzero(residuals.grad) > 0


def test_relational_full_product_and_dephased_controls_are_distinguishable() -> None:
    generated, real, labels = _batch()
    residuals = torch.zeros(6, 16)
    losses = []
    for kernel in (
        RelationalKernel.QUANTUM_FULL,
        RelationalKernel.QUANTUM_PRODUCT,
        RelationalKernel.QUANTUM_DEPHASED,
    ):
        regularizer = TrainableRelationalCoverageReference(
            num_classes=3,
            kernel=kernel,
        )
        regularizer.fit_reference(real, labels)
        losses.append(regularizer(generated, residuals, labels))

    assert not torch.isclose(losses[0], losses[1], atol=1e-7, rtol=0)
    assert not torch.isclose(losses[0], losses[2], atol=1e-7, rtol=0)


def test_relational_wrapper_preserves_kde_component_exactly() -> None:
    generated, real, labels = _batch()
    residuals = torch.zeros(6, 16)
    standalone = ClassConditionalLogKDEReference(num_classes=3)
    guidance = KDERelationalCoverageReference(num_classes=3)
    standalone.fit_reference(real, labels)
    guidance.fit_reference(real, labels)

    expected = standalone(generated, real, labels)
    actual, coverage = guidance.loss_components(generated, residuals, real, labels)
    expected_gradient = torch.autograd.grad(expected, generated, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, generated, retain_graph=True)[0]

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(actual_gradient, expected_gradient)
    assert coverage.isfinite()
    assert sum(parameter.numel() for parameter in guidance.parameters()) == 0
