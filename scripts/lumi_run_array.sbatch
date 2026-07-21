#!/bin/bash
#SBATCH --job-name=vqe-gan-full
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --array=0-9%8

set -euo pipefail

seeds=(42 314 729 2025 4096 5003 65537 9001 12011 16001)
seed=${seeds[${SLURM_ARRAY_TASK_ID}]}

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings/1.0.1

repository_directory=${SLURM_SUBMIT_DIR}
default_container=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif
container_path=${LUMI_CONTAINER_PATH:-${default_container}}
output_root=${VQE_GAN_OUTPUT_ROOT:-runs/lumi-full}
dataset_root=${VQE_GAN_DATASET_ROOT:-data}
classifier_path=${VQE_GAN_CLASSIFIER:-mnist_classifier.pth}
miopen_cache="/tmp/miopen-${USER}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

mkdir -p "${miopen_cache}"
chmod 700 "${miopen_cache}"

cd "${repository_directory}"
export PYTHONPATH="${repository_directory}/src"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export SINGULARITYENV_MIOPEN_USER_DB_PATH="${miopen_cache}"
export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR="${miopen_cache}"

singularity exec "${container_path}" python scripts/run_seed_matrix.py \
    --seed "${seed}" \
    --output-root "${output_root}" \
    --dataset-root "${dataset_root}" \
    --classifier "${classifier_path}" \
    --epochs 10 \
    --evaluation-samples 5000 \
    --device cuda \
    --quantum-device cpu
