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
    parser.add_argument("--gradient-diagnostics-every-steps", type=int, default=0)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")
    return parser


def config_from_arguments(arguments: argparse.Namespace) -> ExperimentConfig:
    variant = ExperimentVariant(arguments.variant)
    regularizer_weight = arguments.regularizer_weight
    if regularizer_weight is None:
        regularizer_weight = 0.0 if variant is ExperimentVariant.NO_REGULARIZER else 1.0
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
