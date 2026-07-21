#!/bin/bash
#SBATCH --job-name=vqe-gan-bench
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:02:00

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings/1.0.1

repository_directory=${SLURM_SUBMIT_DIR}
container_path=$(readlink -f /appl/local/laifs/containers/lumi-multitorch-latest.sif)
miopen_cache="/tmp/miopen-${USER}-${SLURM_JOB_ID}"

mkdir -p "${miopen_cache}"
chmod 700 "${miopen_cache}"

cd "${repository_directory}"
export PYTHONPATH="${repository_directory}/src"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export SINGULARITYENV_MIOPEN_USER_DB_PATH="${miopen_cache}"
export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR="${miopen_cache}"

singularity exec "${container_path}" python - <<'PY'
import json
import platform

import torch

print(
    json.dumps(
        {
            "hostname": platform.node(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
        },
        sort_keys=True,
    )
)
PY

singularity exec "${container_path}" python benchmarks/benchmark_training_step.py \
    --device cuda \
    --quantum-device both \
    --batch-size 64 \
    --warmup 3 \
    --repeats 20 \
    --gradient-ratio 0.1
