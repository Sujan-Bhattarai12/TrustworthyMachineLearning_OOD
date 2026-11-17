#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/asws-mpl-${USER:-user}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$MPLCONFIGDIR"
python - <<'PY'
import json,platform,time
from pathlib import Path
import torch
from scripts.run_cifar10c_study import check
root=Path('results/final_cifar10c')
check(root)
if not torch.cuda.is_available():
    raise SystemExit('CUDA GPU is unavailable. Run inside a GPU allocation with CUDA-enabled PyTorch.')
from importlib.metadata import version
record=dict(started_unix=time.time(),python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0),packages={p:version(p) for p in ['torchvision','transformers','captum','POT','numpy','pandas','scipy','scikit-learn','joblib']})
path=root/f'hpc_environment_{int(time.time())}.json'
path.write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2),flush=True)
PY
# The local writer must be stopped before this process starts.
python -u -m scripts.run_cifar10c_study run --hours 10 2>&1 | tee -a results/final_cifar10c/hpc_run.log
