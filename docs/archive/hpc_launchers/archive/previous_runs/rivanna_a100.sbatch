#!/usr/bin/env bash
#SBATCH --job-name=attribution-final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:30:00
#SBATCH --chdir=/sfs/weka/scratch/cbv8vn/TML_class
#SBATCH --output=/sfs/weka/scratch/cbv8vn/TML_class/attribution-%j.log
# Supply your allocation with: sbatch --account=ALLOCATION hpc/rivanna_a100.sbatch
# If your allocation uses a more specific A100 GRES name, override --gres at submission.
set -euo pipefail
if [[ ! -f .venv-hpc/bin/activate ]]; then
  echo 'Create .venv-hpc with CUDA-enabled PyTorch before submitting.' >&2
  exit 1
fi
source .venv-hpc/bin/activate
bash hpc/run_on_gpu.sh
