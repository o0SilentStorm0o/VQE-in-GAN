"""Evaluate a generator checkpoint with frozen MNIST feature metrics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from vqe_gan.evaluation.run import evaluate_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classifier", default="mnist_classifier.pth")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--output")
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=91_001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--no-download", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = build_parser().parse_args(arguments)
    output = parsed.output or str(Path(parsed.checkpoint).with_name("evaluation.json"))
    result = evaluate_checkpoint(
        parsed.checkpoint,
        parsed.classifier,
        dataset_root=parsed.dataset_root,
        output_path=output,
        samples=parsed.samples,
        batch_size=parsed.batch_size,
        seed=parsed.seed,
        device_name=parsed.device,
        download_dataset=not parsed.no_download,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
