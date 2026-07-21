#!/usr/bin/env python3
"""Calibrate frozen relational-coverage weights from initial development gradients."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

from vqe_gan.config import ExperimentConfig, ExperimentVariant
from vqe_gan.quantum.spec import QuantumCircuitSpec
from vqe_gan.regularizers import KDERelationalCoverageReference, RelationalKernel
from vqe_gan.reproducibility import collect_provenance, seed_everything, write_json
from vqe_gan.runner import _balanced_reference_batch, _build_data_loader, _build_models
from vqe_gan.training import measure_relational_gradient_diagnostics

_SEEDS = (42, 43)
_BATCHES_PER_SEED = 4
_TARGET_RATIO = 0.01
_KDE_WEIGHT = 2e-5
_KERNELS = (
    RelationalKernel.CLASSICAL_PERIODIC_RBF,
    RelationalKernel.QUANTUM_FULL,
    RelationalKernel.QUANTUM_PRODUCT,
    RelationalKernel.QUANTUM_DEPHASED,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--reference-samples-per-class", type=int, default=1_024)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError(f"calibration output already exists: {output}")
    if arguments.reference_samples_per_class <= 0:
        raise ValueError("reference-samples-per-class must be positive")

    repository_root = Path(__file__).resolve().parents[1]
    measurements: dict[str, list[dict[str, object]]] = {
        RelationalKernel.CLASSICAL_PERIODIC_RBF.value: [],
        RelationalKernel.QUANTUM_FULL.value: [],
    }
    structural_checks: list[dict[str, object]] = []
    for seed in _SEEDS:
        seed_everything(seed)
        config = ExperimentConfig(
            run_name=f"relational-calibration-{seed}",
            dataset_root=arguments.dataset_root,
            variant=ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE,
            seed=seed,
            device="cpu",
            quantum_device="cpu",
            batch_size=64,
            regularizer_weight=_KDE_WEIGHT,
            coverage_weight=1.0,
            contrastive_reference_samples_per_class=(
                arguments.reference_samples_per_class
            ),
            download_dataset=arguments.download,
        )
        device = torch.device("cpu")
        data_loader = _build_data_loader(config, device)
        generator, discriminator, _ = _build_models(config, device, device)
        reference_images, reference_labels, actual_reference_count = (
            _balanced_reference_batch(
                data_loader.dataset,
                num_classes=config.num_classes,
                samples_per_class=arguments.reference_samples_per_class,
            )
        )
        regularizers = {
            kernel: _build_regularizer(config, kernel) for kernel in _KERNELS
        }
        for regularizer in regularizers.values():
            regularizer.fit_reference(reference_images, reference_labels)
        _assert_identical_kde_references(regularizers)

        batch_iterator = iter(data_loader)
        for batch_index in range(_BATCHES_PER_SEED):
            real_images, class_labels = next(batch_iterator)
            noise = torch.randn(real_images.shape[0], config.latent_dim)
            diagnostics = {}
            for kernel in (
                RelationalKernel.CLASSICAL_PERIODIC_RBF,
                RelationalKernel.QUANTUM_FULL,
            ):
                diagnostic = measure_relational_gradient_diagnostics(
                    generator,
                    discriminator,
                    regularizers[kernel],
                    real_images,
                    noise,
                    class_labels,
                    kde_weight=_KDE_WEIGHT,
                    coverage_weight=1.0,
                )
                if diagnostic.coverage_shared_norm <= 0:
                    raise RuntimeError(f"zero shared coverage gradient for {kernel.value}")
                if diagnostic.coverage_angle_head_norm <= 0:
                    raise RuntimeError(f"zero angle-head gradient for {kernel.value}")
                raw_ratio = diagnostic.weighted_coverage_to_gan_norm_ratio
                record = {
                    "seed": seed,
                    "batch_index": batch_index,
                    "class_counts": torch.bincount(
                        class_labels,
                        minlength=config.num_classes,
                    ).tolist(),
                    "candidate_weight": _TARGET_RATIO / raw_ratio,
                    **asdict(diagnostic),
                }
                measurements[kernel.value].append(record)
                diagnostics[kernel] = diagnostic

            classical_kde = diagnostics[
                RelationalKernel.CLASSICAL_PERIODIC_RBF
            ].kde_loss
            quantum_kde = diagnostics[RelationalKernel.QUANTUM_FULL].kde_loss
            if classical_kde != quantum_kde:
                raise RuntimeError("matched variants changed the exact KDE loss")

            if batch_index == 0:
                structural_checks.append(
                    {
                        "seed": seed,
                        "reference_samples_per_class": actual_reference_count,
                        "zero_angle_residual": _angle_residual_is_exactly_zero(
                            generator,
                            noise,
                            class_labels,
                        ),
                        "kde_loss_exact_between_matched_variants": True,
                        "ablations": _measure_ablation_differences(
                            generator,
                            regularizers,
                            real_images,
                            noise,
                            class_labels,
                        ),
                    }
                )

    suggested_weights = {
        ExperimentVariant.CLASSICAL_KDE_RELATIONAL_COVERAGE.value: statistics.median(
            float(record["candidate_weight"])
            for record in measurements[RelationalKernel.CLASSICAL_PERIODIC_RBF.value]
        ),
        ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE.value: statistics.median(
            float(record["candidate_weight"])
            for record in measurements[RelationalKernel.QUANTUM_FULL.value]
        ),
    }
    result = {
        "schema_version": 1,
        "protocol": "docs/trainable_relational_coverage_protocol.md",
        "seeds": list(_SEEDS),
        "batches_per_seed": _BATCHES_PER_SEED,
        "target_initial_shared_gradient_ratio": _TARGET_RATIO,
        "kde_weight": _KDE_WEIGHT,
        "reference_samples_per_class": arguments.reference_samples_per_class,
        "measurements": measurements,
        "suggested_weights": suggested_weights,
        "structural_checks": structural_checks,
        "provenance": collect_provenance(repository_root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _build_regularizer(
    config: ExperimentConfig,
    kernel: RelationalKernel,
) -> KDERelationalCoverageReference:
    return KDERelationalCoverageReference(
        QuantumCircuitSpec(num_qubits=config.num_qubits, reps=config.ansatz_reps),
        num_classes=config.num_classes,
        execution_device="cpu",
        angle_scale=config.contrastive_angle_scale,
        angle_residual_fraction=config.angle_residual_fraction,
        coverage_temperature=config.coverage_temperature,
        kde_sigma_squared=config.kde_sigma_squared,
        kde_temperature=config.kde_temperature,
        kernel=kernel,
    )


def _assert_identical_kde_references(
    regularizers: dict[RelationalKernel, KDERelationalCoverageReference],
) -> None:
    expected = regularizers[RelationalKernel.QUANTUM_FULL].classical.reference_features
    for regularizer in regularizers.values():
        if not torch.equal(expected, regularizer.classical.reference_features):
            raise RuntimeError("matched variants received different KDE references")


def _angle_residual_is_exactly_zero(
    generator: torch.nn.Module,
    noise: torch.Tensor,
    class_labels: torch.Tensor,
) -> bool:
    with torch.no_grad():
        _, residuals = generator.forward_with_angles(noise, class_labels)
    return bool(torch.count_nonzero(residuals).item() == 0)


def _measure_ablation_differences(
    generator: torch.nn.Module,
    regularizers: dict[RelationalKernel, KDERelationalCoverageReference],
    real_images: torch.Tensor,
    noise: torch.Tensor,
    class_labels: torch.Tensor,
) -> dict[str, object]:
    losses_and_gradients = {
        kernel: _coverage_loss_and_shared_gradients(
            generator,
            regularizers[kernel],
            real_images,
            noise,
            class_labels,
        )
        for kernel in (
            RelationalKernel.QUANTUM_FULL,
            RelationalKernel.QUANTUM_PRODUCT,
            RelationalKernel.QUANTUM_DEPHASED,
        )
    }
    full_loss, full_gradients = losses_and_gradients[RelationalKernel.QUANTUM_FULL]
    result = {}
    for kernel in (
        RelationalKernel.QUANTUM_PRODUCT,
        RelationalKernel.QUANTUM_DEPHASED,
    ):
        control_loss, control_gradients = losses_and_gradients[kernel]
        difference = tuple(
            full - control
            for full, control in zip(full_gradients, control_gradients, strict=True)
        )
        result[kernel.value] = {
            "full_loss": full_loss,
            "control_loss": control_loss,
            "loss_difference": full_loss - control_loss,
            "gradient_difference_norm": _gradient_norm(difference),
            "gradient_cosine": _gradient_cosine(full_gradients, control_gradients),
        }
    return result


def _coverage_loss_and_shared_gradients(
    generator: torch.nn.Module,
    regularizer: KDERelationalCoverageReference,
    real_images: torch.Tensor,
    noise: torch.Tensor,
    class_labels: torch.Tensor,
) -> tuple[float, tuple[torch.Tensor, ...]]:
    generator_was_training = generator.training
    generator.eval()
    try:
        images, residuals = generator.forward_with_angles(noise, class_labels)
        _, loss = regularizer.loss_components(
            images,
            residuals,
            real_images,
            class_labels,
        )
        parameters = (
            *generator.label_embedding.parameters(),
            *generator.input_projection.parameters(),
            *generator.image_decoder.parameters(),
        )
        raw_gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        gradients = tuple(
            gradient if gradient is not None else torch.zeros_like(parameter)
            for gradient, parameter in zip(raw_gradients, parameters, strict=True)
        )
    finally:
        generator.train(generator_was_training)
    return loss.detach().item(), gradients


def _gradient_norm(gradients: tuple[torch.Tensor, ...]) -> float:
    return math.sqrt(sum(gradient.square().sum().item() for gradient in gradients))


def _gradient_cosine(
    first: tuple[torch.Tensor, ...],
    second: tuple[torch.Tensor, ...],
) -> float | None:
    denominator = _gradient_norm(first) * _gradient_norm(second)
    if denominator == 0:
        return None
    dot = sum(
        (left * right).sum().item()
        for left, right in zip(first, second, strict=True)
    )
    return dot / denominator


if __name__ == "__main__":
    main()
