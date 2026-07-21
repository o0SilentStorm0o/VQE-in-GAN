"""Measure end-to-end training and evaluation runtime for the frozen protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vqe_gan.config import ExperimentConfig, ExperimentVariant
from vqe_gan.evaluation.run import evaluate_checkpoint
from vqe_gan.runner import run_experiment


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--classifier", default="mnist_classifier.pth")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--evaluation-samples", type=int, default=5_000)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--quantum-device", choices=("same", "cpu"), default="cpu")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    variants = list(ExperimentVariant)
    runtimes: dict[str, dict[str, float | int]] = {}
    checkpoints: dict[ExperimentVariant, Path] = {}
    benchmark_start = time.perf_counter()

    for variant in variants:
        run_name = f"runtime-{variant.value}"
        config = ExperimentConfig(
            run_name=run_name,
            output_root=arguments.output_root,
            dataset_root=arguments.dataset_root,
            variant=variant,
            seed=42,
            device=arguments.device,
            quantum_device=arguments.quantum_device,
            epochs=1,
            max_steps=arguments.steps,
            regularizer_weight=0.0 if variant is ExperimentVariant.NO_REGULARIZER else 1.0,
            regularizer_gradient_ratio=(
                None if variant is ExperimentVariant.NO_REGULARIZER else 0.1
            ),
            download_dataset=False,
        )
        start = time.perf_counter()
        summary = run_experiment(config)
        elapsed = time.perf_counter() - start
        completed_steps = int(summary["completed_steps"])
        runtimes[variant.value] = {
            "completed_steps": completed_steps,
            "elapsed_seconds": elapsed,
            "seconds_per_step": elapsed / completed_steps,
        }
        checkpoints[variant] = config.output_directory / "checkpoint-final.pt"

    evaluation_start = time.perf_counter()
    evaluation_output = Path(arguments.output_root) / "evaluation-runtime.json"
    evaluate_checkpoint(
        checkpoints[ExperimentVariant.NO_REGULARIZER],
        arguments.classifier,
        dataset_root=arguments.dataset_root,
        output_path=evaluation_output,
        samples=arguments.evaluation_samples,
        batch_size=128,
        device_name=arguments.device,
        download_dataset=False,
    )
    evaluation_seconds = time.perf_counter() - evaluation_start

    result = {
        "device": arguments.device,
        "quantum_device": arguments.quantum_device,
        "steps_per_run": arguments.steps,
        "training": runtimes,
        "evaluation_samples": arguments.evaluation_samples,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": time.perf_counter() - benchmark_start,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
