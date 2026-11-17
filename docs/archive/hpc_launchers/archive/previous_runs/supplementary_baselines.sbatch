#!/bin/bash
#SBATCH --job-name=entropy-energy
#SBATCH --account=uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/sfs/weka/scratch/cbv8vn/TML_class
#SBATCH --output=entropy-energy-%j.log
set -euo pipefail
source .venv-hpc/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
python -u -m scripts.evaluate_supplementary_baselines run
