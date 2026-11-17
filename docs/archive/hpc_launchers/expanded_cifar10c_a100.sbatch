#!/usr/bin/env bash
#SBATCH --job-name=lgd-expanded
#SBATCH --account=uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/sfs/weka/scratch/cbv8vn/TML_class
#SBATCH --output=lgd-expanded-%j.log
set -euo pipefail
source .venv-hpc/bin/activate
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/lgd-expanded-mpl-${USER:-user}"
mkdir -p "$MPLCONFIGDIR"
python - <<'PY'
import torch,json,time,platform
from pathlib import Path
if not torch.cuda.is_available():raise SystemExit('CUDA GPU unavailable.')
p=Path('results/expanded_cifar10c_5000dev_2000test')
record=dict(started_unix=time.time(),python=platform.python_version(),torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
(p/f'hpc_environment_{int(time.time())}.json').write_text(json.dumps(record,indent=2))
print(record,flush=True)
PY
python -u -m scripts.run_expanded_cifar10c_study run --hours 47.5 2>&1 | tee -a results/expanded_cifar10c_5000dev_2000test/hpc_run.log
