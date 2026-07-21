#!/usr/bin/env python3
"""Run one frozen full-trajectory reference-budget phase on development seeds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vqe_gan.config import CoverageBudgetMode, ExperimentConfig, ExperimentVariant
from vqe_gan.evaluation.run import evaluate_checkpoint
from vqe_gan.reproducibility import file_sha256, write_json
from vqe_gan.runner import run_experiment

SEEDS = (42, 43)
VARIANTS = (
    ExperimentVariant.QUANTUM_KDE_RELATIONAL_COVERAGE,
    ExperimentVariant.QUANTUM_KDE_RELATIONAL_PRODUCT,
    ExperimentVariant.QUANTUM_KDE_RELATIONAL_DEPHASED,
)
REFERENCE_VARIANT = VARIANTS[0]
MAX_STEPS = 200
EVALUATION_SAMPLES = 5_000
EVALUATION_SEED = 91_001
REFERENCE_SAMPLES_PER_CLASS = 1_024
KDE_WEIGHT = 2e-5
COVERAGE_WEIGHT = REFERENCE_VARIANT.default_coverage_weight


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("a", "b"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--classifier", default="mnist_classifier.pth")
    return parser.parse_args()


def run_name(phase: str, variant: ExperimentVariant, seed: int) -> str:
    return f"reference-budget-{phase}-{variant.value}-seed-{seed}"


def main() -> None:
    arguments = parse_arguments()
    output_root = Path(arguments.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    matrix_path = output_root / f"reference-budget-phase-{arguments.phase}.json"
    if matrix_path.exists():
        raise FileExistsError(f"phase matrix already exists: {matrix_path}")
    if arguments.phase == "b":
        phase_a_audit = output_root / "reference-budget-phase-a-audit.json"
        if not phase_a_audit.is_file():
            raise FileNotFoundError("Phase B requires the completed Phase A audit")

    run_directories: dict[tuple[int, ExperimentVariant], Path] = {}
    training_seconds: dict[str, float] = {}
    evaluation_seconds: dict[str, float] = {}
    phase_start = time.perf_counter()

    # Outcomes remain unavailable until every training branch in the phase is complete.
    for seed in SEEDS:
        reference_directory = output_root / run_name(arguments.phase, REFERENCE_VARIANT, seed)
        for variant in VARIANTS:
            name = run_name(arguments.phase, variant, seed)
            schedule = (
                str(reference_directory / "coverage-budget.json")
                if variant is not REFERENCE_VARIANT
                else None
            )
            config = ExperimentConfig(
                run_name=name,
                output_root=str(output_root),
                dataset_root=arguments.dataset_root,
                variant=variant,
                seed=seed,
                device="cpu",
                quantum_device="cpu",
                epochs=1,
                max_steps=MAX_STEPS,
                batch_size=64,
                num_workers=0,
                regularizer_weight=KDE_WEIGHT,
                contrastive_reference_samples_per_class=(REFERENCE_SAMPLES_PER_CLASS),
                coverage_weight=COVERAGE_WEIGHT,
                coverage_budget_mode=(
                    CoverageBudgetMode.RECORD
                    if variant is REFERENCE_VARIANT
                    else CoverageBudgetMode.REPLAY
                ),
                coverage_budget_schedule=schedule,
                freeze_angle_head=arguments.phase == "a",
                match_angle_head_budget=arguments.phase == "b",
                log_every_steps=1,
                checkpoint_every_steps=0,
                download_dataset=False,
            )
            start = time.perf_counter()
            run_experiment(config)
            training_seconds[f"{seed}:{variant.value}"] = time.perf_counter() - start
            run_directories[(seed, variant)] = config.output_directory.resolve()

    for seed in SEEDS:
        for variant in VARIANTS:
            directory = run_directories[(seed, variant)]
            start = time.perf_counter()
            evaluate_checkpoint(
                directory / "checkpoint-final.pt",
                arguments.classifier,
                dataset_root=arguments.dataset_root,
                output_path=directory / "evaluation.json",
                samples=EVALUATION_SAMPLES,
                batch_size=128,
                seed=EVALUATION_SEED,
                device_name="cpu",
                download_dataset=False,
            )
            evaluation_seconds[f"{seed}:{variant.value}"] = time.perf_counter() - start

    runs = []
    for seed in SEEDS:
        for variant in VARIANTS:
            directory = run_directories[(seed, variant)]
            schedule_path = (
                directory / "coverage-budget.json"
                if variant is REFERENCE_VARIANT
                else Path(
                    json.loads((directory / "config.json").read_text(encoding="utf-8"))[
                        "coverage_budget_schedule"
                    ]
                )
            )
            runs.append(
                {
                    "seed": seed,
                    "variant": variant.value,
                    "run_directory": str(directory),
                    "checkpoint_sha256": file_sha256(directory / "checkpoint-final.pt"),
                    "evaluation_sha256": file_sha256(directory / "evaluation.json"),
                    "coverage_budget_schedule": str(schedule_path),
                    "coverage_budget_schedule_sha256": file_sha256(schedule_path),
                    "training_seconds": training_seconds[f"{seed}:{variant.value}"],
                    "evaluation_seconds": evaluation_seconds[f"{seed}:{variant.value}"],
                }
            )
    result = {
        "schema_version": 1,
        "status": "development_complete_pending_audit",
        "protocol": "docs/reference_budget_relational_protocol.md",
        "phase": arguments.phase,
        "phase_a_audit": (
            str(output_root / "reference-budget-phase-a-audit.json")
            if arguments.phase == "b"
            else None
        ),
        "seeds": list(SEEDS),
        "max_steps": MAX_STEPS,
        "evaluation_samples": EVALUATION_SAMPLES,
        "evaluation_seed": EVALUATION_SEED,
        "device": "cpu",
        "runs": runs,
        "total_seconds": time.perf_counter() - phase_start,
    }
    write_json(matrix_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
