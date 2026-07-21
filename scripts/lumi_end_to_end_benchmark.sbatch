#!/bin/bash
#SBATCH --job-name=vqe-gan-e2e
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
output_root="runtime-runs-${SLURM_JOB_ID}"
miopen_cache="/tmp/miopen-${USER}-${SLURM_JOB_ID}"

mkdir -p "${miopen_cache}"
chmod 700 "${miopen_cache}"

cd "${repository_directory}"
export PYTHONPATH="${repository_directory}/src"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export SINGULARITYENV_MIOPEN_USER_DB_PATH="${miopen_cache}"
export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR="${miopen_cache}"

singularity exec "${container_path}" python benchmarks/benchmark_experiment_runtime.py \
    --dataset-root data \
    --classifier mnist_classifier.pth \
    --output-root "${output_root}" \
    --steps 300 \
    --evaluation-samples 5000 \
    --device cuda \
    --quantum-device cpu
