#!/bin/bash
#SBATCH --job-name=entropy-energy-expanded
#SBATCH --account=uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/sfs/weka/scratch/cbv8vn/TML_class
#SBATCH --output=entropy-energy-expanded-%j.log
set -euo pipefail
source .venv-hpc/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
python -u -m scripts.evaluate_expanded_supplementary_baselines run --root results/expanded_cifar10c_5000dev_2000test
