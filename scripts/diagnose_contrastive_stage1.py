"""Run the fixed post-hoc diagnostic for failed contrastive Stage 1 checkpoint pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vqe_gan.analysis import diagnose_contrastive_failure
from vqe_gan.reproducibility import write_json


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classical-checkpoints", nargs="+", required=True)
    parser.add_argument("--hybrid-checkpoints", nargs="+", required=True)
    parser.add_argument("--classifier", default="mnist_classifier.pth")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-samples", type=int, default=5_000)
    parser.add_argument("--score-samples", type=int, default=1_000)
    parser.add_argument("--gradient-samples", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=91_001)
    parser.add_argument("--gradient-seed", type=int, default=123_456)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    result = diagnose_contrastive_failure(
        arguments.classical_checkpoints,
        arguments.hybrid_checkpoints,
        arguments.classifier,
        dataset_root=arguments.dataset_root,
        evaluation_samples=arguments.evaluation_samples,
        score_samples=arguments.score_samples,
        gradient_samples=arguments.gradient_samples,
        evaluation_seed=arguments.evaluation_seed,
        gradient_seed=arguments.gradient_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "cross_seed_summary": result["cross_seed_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
