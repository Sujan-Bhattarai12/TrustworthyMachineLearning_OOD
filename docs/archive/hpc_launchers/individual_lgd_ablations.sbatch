#!/bin/bash
#SBATCH --job-name=lgd-individual-ablations
#SBATCH --account=uqgroup
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/sfs/weka/scratch/cbv8vn/TML_class
#SBATCH --output=lgd-individual-%j.log
set -euo pipefail
source .venv-hpc/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
python -u -m scripts.evaluate_individual_lgd run
