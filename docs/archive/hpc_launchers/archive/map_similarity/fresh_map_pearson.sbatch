#!/bin/bash
#SBATCH --job-name=pearson-fresh200
#SBATCH --account=uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --chdir=/sfs/weka/scratch/cbv8vn/TML_class
#SBATCH --output=pearson-fresh200-%j.log
set -euo pipefail
source .venv-hpc/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
python -u -m scripts.collect_attribution_metrics --dataset cifar10c --classifier resnet50 --checkpoint checkpoints/huggingface/resnet50 --manifest results/fresh_map_pearson_200/fresh_test_manifest.csv --split test --all-clean-correct --corruptions gaussian_blur --ig-steps 64 --save-maps --resume --output-dir results/fresh_map_pearson_200/collection
python -u -m scripts.evaluate_fresh_map_pearson
