"""Command-line entry point for corrected experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from vqe_gan.config import ExperimentConfig, ExperimentVariant
from vqe_gan.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in ExperimentVariant],
        default=ExperimentVariant.QUANTUM_CONTRASTIVE.value,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--quantum-device",
        choices=("auto", "same", "cpu"),
        default="auto",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--regularizer-weight", type=float)
    parser.add_argument("--regularizer-temperature", type=float, default=1.0)
    parser.add_argument("--regularizer-gradient-ratio", type=float)
    parser.add_argument("--modular-angle-scale", type=float, default=0.125)
    parser.add_argument("--modular-depolarization", type=float, default=0.01)
    parser.add_argument("--density-mmd-angle-scale", type=float, default=0.25)
    parser.add_argument("--rbf-sigma-squared", type=float, default=1.213)
    parser.add_argument("--coherence-gradient-ratio", type=float, default=0.05)
    parser.add_argument("--reference-samples-per-class", type=int, default=256)
    parser.add_argument("--contrastive-reference-samples-per-class", type=int, default=1024)
    parser.add_argument("--contrastive-angle-scale", type=float, default=0.75)
    parser.add_argument("--contrastive-quantum-temperature", type=float, default=0.04)
    parser.add_argument("--kde-sigma-squared", type=float, default=0.03125)
    parser.add_argument("--kde-temperature", type=float, default=0.75)
    parser.add_argument("--quantum-mixture-weight", type=float, default=0.05)
    parser.add_argument("--coverage-weight", type=float)
    parser.add_argument("--coverage-temperature", type=float, default=0.10)
    parser.add_argument("--angle-residual-fraction", type=float, default=0.10)
    parser.add_argument("--gradient-diagnostics-every-steps", type=int, default=0)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")
    return parser


def config_from_arguments(arguments: argparse.Namespace) -> ExperimentConfig:
    variant = ExperimentVariant(arguments.variant)
    regularizer_weight = arguments.regularizer_weight
    if regularizer_weight is None:
        regularizer_weight = variant.default_regularizer_weight
    coverage_weight = arguments.coverage_weight
    if coverage_weight is None:
        coverage_weight = variant.default_coverage_weight
    return ExperimentConfig(
        run_name=arguments.run_name,
        output_root=arguments.output_root,
        dataset_root=arguments.dataset_root,
        variant=variant,
        seed=arguments.seed,
        device=arguments.device,
        quantum_device=arguments.quantum_device,
        epochs=arguments.epochs,
        max_steps=arguments.max_steps,
        dataset_limit=arguments.dataset_limit,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        regularizer_weight=regularizer_weight,
        regularizer_temperature=arguments.regularizer_temperature,
        regularizer_gradient_ratio=arguments.regularizer_gradient_ratio,
        modular_angle_scale=arguments.modular_angle_scale,
        modular_depolarization=arguments.modular_depolarization,
        density_mmd_angle_scale=arguments.density_mmd_angle_scale,
        rbf_sigma_squared=arguments.rbf_sigma_squared,
        coherence_gradient_ratio=arguments.coherence_gradient_ratio,
        reference_samples_per_class=arguments.reference_samples_per_class,
        contrastive_reference_samples_per_class=(
            arguments.contrastive_reference_samples_per_class
        ),
        contrastive_angle_scale=arguments.contrastive_angle_scale,
        contrastive_quantum_temperature=arguments.contrastive_quantum_temperature,
        kde_sigma_squared=arguments.kde_sigma_squared,
        kde_temperature=arguments.kde_temperature,
        quantum_mixture_weight=arguments.quantum_mixture_weight,
        coverage_weight=coverage_weight,
        coverage_temperature=arguments.coverage_temperature,
        angle_residual_fraction=arguments.angle_residual_fraction,
        gradient_diagnostics_every_steps=arguments.gradient_diagnostics_every_steps,
        checkpoint_every_steps=arguments.checkpoint_every_steps,
        download_dataset=not arguments.no_download,
    )


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = build_parser().parse_args(arguments)
    summary = run_experiment(config_from_arguments(parsed))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
