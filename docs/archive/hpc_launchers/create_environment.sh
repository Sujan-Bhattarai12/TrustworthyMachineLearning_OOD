#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Run in a login shell after module load miniforge/26.9.4.
if ! command -v conda >/dev/null; then
  echo 'Load miniforge/26.9.4 before running this script.' >&2
  exit 1
fi
export CONDA_PKGS_DIRS="$PWD/.conda-package-cache"
if [[ ! -x .python-hpc/bin/python ]]; then
  conda create --yes --prefix "$PWD/.python-hpc" --override-channels -c conda-forge python=3.11 pip
fi
if [[ ! -x .venv-hpc/bin/python ]]; then
  .python-hpc/bin/python -m venv .venv-hpc
fi
source .venv-hpc/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir 'torch==2.14.1' 'torchvision==0.29.1' --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir -r hpc/requirements.txt
python -m pip check
python - <<'PY'
import json,platform
from pathlib import Path
from importlib.metadata import version
import torch
from scripts.run_cifar10c_study import check
check(Path('results/final_cifar10c'))
packages=['torch','torchvision','transformers','captum','POT','numpy','pandas','scipy','scikit-learn','joblib','safetensors','matplotlib']
record=dict(python=platform.python_version(),cuda_build=torch.version.cuda,packages={p:version(p) for p in packages})
Path('hpc/installed_environment.json').write_text(json.dumps(record,indent=2))
assert torch.version.cuda is not None,'Expected a CUDA-enabled PyTorch build.'
print(json.dumps(record,indent=2))
print('Environment installed; GPU availability must be tested in a Slurm allocation.')
PY
