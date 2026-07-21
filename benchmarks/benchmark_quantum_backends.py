"""Measure forward/backward latency for the parity-tested energy backends."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import time
from collections.abc import Callable

import torch
from torch import nn

from vqe_gan.quantum.qiskit_backend import QiskitReferenceEnergy
from vqe_gan.quantum.torch_backend import TorchStatevectorEnergy


def measure(
    factory: Callable[[], nn.Module],
    batch_size: int,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    backend = factory()
    generator = torch.Generator().manual_seed(2025 + batch_size)
    base_angles = torch.randn(batch_size, backend.num_parameters, generator=generator)
    labels = torch.arange(batch_size, dtype=torch.long) % 10

    def step() -> None:
        angles = base_angles.clone().requires_grad_(True)
        backend(angles, labels).mean().backward()

    for _ in range(warmup):
        step()

    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        step()
        timings.append((time.perf_counter() - started) * 1_000)

    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.mean(timings),
        "min_ms": min(timings),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("torch", "qiskit-reverse", "qiskit-param-shift", "all"),
        default="all",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32, 64])
    args = parser.parse_args()

    measurements: dict[str, list[dict[str, float | int]]] = {}
    results: dict[str, object] = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "qiskit": importlib.metadata.version("qiskit"),
            "qiskit_machine_learning": importlib.metadata.version("qiskit-machine-learning"),
        },
        "measurements": measurements,
    }
    if args.backend in {"torch", "all"}:
        measurements["torch"] = [
            measure(TorchStatevectorEnergy, size, warmup=10, iterations=100)
            for size in args.batch_sizes
        ]
    if args.backend in {"qiskit-reverse", "all"}:
        measurements["qiskit_reverse"] = [
            measure(QiskitReferenceEnergy, size, warmup=1, iterations=3)
            for size in args.batch_sizes
        ]
    if args.backend in {"qiskit-param-shift", "all"}:
        measurements["qiskit_param_shift"] = [
            measure(
                lambda: QiskitReferenceEnergy(reverse_gradient=False),
                size,
                warmup=1,
                iterations=3,
            )
            for size in args.batch_sizes
        ]

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
