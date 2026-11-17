#!/usr/bin/env bash
#SBATCH --job-name=attribution-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --chdir=/sfs/weka/scratch/cbv8vn/TML_class
#SBATCH --output=/sfs/weka/scratch/cbv8vn/TML_class/gpu-check-%j.log
set -euo pipefail
source .venv-hpc/bin/activate
export MPLCONFIGDIR="${TMPDIR:-/tmp}/asws-mpl-check-${USER}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
python - <<'PY'
import json,time
from pathlib import Path
import numpy as np
import torch
from captum.attr import IntegratedGradients
from src.classifiers import load_published_classifier,attribution_layer
from src.cifar10c_loader import CIFAR10CLoader
from scripts.collect_attribution_metrics import compute_ig,compute_cam,asws_components
from scripts.archive.legacy_entrypoints.run_cifar10c_study import check
root=Path('results/final_cifar10c');check(root)
assert torch.cuda.is_available(),'No CUDA device available'
print('GPU:',torch.cuda.get_device_name(0),flush=True)
loader=CIFAR10CLoader(root/'development_manifest.csv')
image=int(loader.manifest[0]['image_id']);result=[]
for name in ['resnet50','vit_b_16','swin_t']:
    start=time.time()
    model=load_published_classifier(Path('checkpoints/huggingface')/name,name,'cuda')
    clean=loader.get_image(image,'test','cuda')
    altered=loader.get_corrupted_image(image,'test','gaussian_noise',1,'cuda')
    with torch.no_grad():target=int(model(clean).argmax(1))
    layer,reshape=attribution_layer(model,name)
    ig=IntegratedGradients(model)
    clean_ig=compute_ig(ig,clean,target,64);other_ig=compute_ig(ig,altered,target,64)
    clean_cam=compute_cam(model,clean,target,layer,reshape);other_cam=compute_cam(model,altered,target,layer,reshape)
    assert all(np.isfinite(m).all() for m in [clean_ig,other_ig,clean_cam,other_cam])
    torch.cuda.synchronize()
    record=dict(classifier=name,seconds=time.time()-start,ig_steps=64,ig=asws_components(clean_ig,other_ig),gradcam=asws_components(clean_cam,other_cam))
    print(json.dumps(record),flush=True);result.append(record)
    del model,ig,clean,altered
    torch.cuda.empty_cache()
(root/'gpu_preflight').mkdir(exist_ok=True)
(root/'gpu_preflight/results.json').write_text(json.dumps(result,indent=2))
print('A100_CHECK_PASSED',flush=True)
PY
