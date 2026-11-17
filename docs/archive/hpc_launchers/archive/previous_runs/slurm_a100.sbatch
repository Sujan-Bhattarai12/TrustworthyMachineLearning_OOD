#!/usr/bin/env bash
# TEMPLATE: verify account, partition and GPU resource syntax with your HPC.
#SBATCH --job-name=attribution-final
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:30:00
#SBATCH --output=attribution-%j.log
set -euo pipefail
# Submit from the project root, after activating your CUDA Python environment.
bash hpc/run_on_gpu.sh
