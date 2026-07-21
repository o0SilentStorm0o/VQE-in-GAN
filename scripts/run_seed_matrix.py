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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--classifier", default="mnist_classifier.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--evaluation-samples", type=int, default=5_000)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--quantum-device", choices=("same", "cpu"), default="cpu")
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

    for variant in ExperimentVariant:
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
            regularizer_weight=0.0 if variant is ExperimentVariant.NO_REGULARIZER else 1.0,
            regularizer_gradient_ratio=(
                None if variant is ExperimentVariant.NO_REGULARIZER else 0.1
            ),
            download_dataset=False,
        )
        synchronize(arguments.device)
        start = time.perf_counter()
        run_experiment(config)
        synchronize(arguments.device)
        training_seconds[variant.value] = time.perf_counter() - start
        run_directories[variant] = config.output_directory

    for variant in ExperimentVariant:
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
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": time.perf_counter() - matrix_start,
    }
    summary_path = output_root / f"matrix-seed-{arguments.seed}.json"
    write_json(summary_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
