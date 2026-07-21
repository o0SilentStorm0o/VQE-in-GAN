"""Run and evaluate every frozen ablation variant for one paired seed."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vqe_gan.config import ExperimentConfig, ExperimentVariant
from vqe_gan.evaluation.run import evaluate_checkpoint
from vqe_gan.reproducibility import write_json
from vqe_gan.runner import run_experiment

DEFAULT_VARIANTS = (
    ExperimentVariant.NO_REGULARIZER,
    ExperimentVariant.QUANTUM_CONTRASTIVE,
    ExperimentVariant.CLASSICAL_PROTOTYPE,
    ExperimentVariant.QUANTUM_PERMUTED,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--classifier", default="mnist_classifier.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--evaluation-samples", type=int, default=5_000)
    parser.add_argument("--coverage-weight", type=float)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--quantum-device", choices=("same", "cpu"), default="cpu")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[variant.value for variant in ExperimentVariant],
        default=[variant.value for variant in DEFAULT_VARIANTS],
        help="Explicit variant subset; defaults to the original four-way ablation.",
    )
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main() -> None:
    arguments = parse_arguments()
    output_root = Path(arguments.output_root)
    run_directories: dict[ExperimentVariant, Path] = {}
    training_seconds: dict[str, float] = {}
    evaluation_seconds: dict[str, float] = {}
    matrix_start = time.perf_counter()
    variants = [ExperimentVariant(value) for value in arguments.variants]

    for variant in variants:
        run_name = f"{variant.value}-seed-{arguments.seed}"
        config = ExperimentConfig(
            run_name=run_name,
            output_root=str(output_root),
            dataset_root=arguments.dataset_root,
            variant=variant,
            seed=arguments.seed,
            device=arguments.device,
            quantum_device=arguments.quantum_device,
            epochs=arguments.epochs,
            max_steps=arguments.max_steps,
            regularizer_weight=variant.default_regularizer_weight,
            regularizer_gradient_ratio=(
                None
                if variant is ExperimentVariant.NO_REGULARIZER
                or variant.uses_contrastive_reference
                else 0.1
            ),
            coverage_weight=(
                arguments.coverage_weight
                if arguments.coverage_weight is not None
                and variant.uses_relational_coverage
                else variant.default_coverage_weight
            ),
            download_dataset=False,
        )
        synchronize(arguments.device)
        start = time.perf_counter()
        run_experiment(config)
        synchronize(arguments.device)
        training_seconds[variant.value] = time.perf_counter() - start
        run_directories[variant] = config.output_directory

    for variant in variants:
        run_directory = run_directories[variant]
        synchronize(arguments.device)
        start = time.perf_counter()
        evaluate_checkpoint(
            run_directory / "checkpoint-final.pt",
            arguments.classifier,
            dataset_root=arguments.dataset_root,
            output_path=run_directory / "evaluation.json",
            samples=arguments.evaluation_samples,
            batch_size=128,
            device_name=arguments.device,
            download_dataset=False,
        )
        synchronize(arguments.device)
        evaluation_seconds[variant.value] = time.perf_counter() - start

    result = {
        "seed": arguments.seed,
        "epochs": arguments.epochs,
        "evaluation_samples": arguments.evaluation_samples,
        "device": arguments.device,
        "quantum_device": arguments.quantum_device,
        "variants": [variant.value for variant in variants],
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": time.perf_counter() - matrix_start,
    }
    summary_path = output_root / f"matrix-seed-{arguments.seed}.json"
    write_json(summary_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
