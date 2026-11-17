> Updated protocol: use official CIFAR-10-C files and the new `results/final_cifar10c/` run. The stopped custom-corruption run is superseded; its 85 images cannot be reused as official benchmark observations. Sync the updated code, manifests and `data/CIFAR-10-C/` before submission.

# UVA Rivanna/Afton transfer

Destination: `/sfs/weka/scratch/cbv8vn/TML_class`. Commands assume the computing ID is `cbv8vn`, inferred from the supplied path. The current UVA SSH hostname is `login.hpc.virginia.edu` (https://rc.virginia.edu/request-manage/ssz-login). Use UVA VPN if required by your connection; complete SSH/Duo authentication yourself.

## First copy — Mac terminal

```bash
cd '/Users/sujanbhattarai/Desktop/PhD Classes/Trustworthy Machine learning/final_project'
ssh cbv8vn@login.hpc.virginia.edu 'mkdir -p /sfs/weka/scratch/cbv8vn/TML_class'
rsync -avh --progress --exclude='.venv/' --exclude='.venv-hpc/' --exclude='.git/' --exclude='__pycache__/' --exclude='.DS_Store' ./ cbv8vn@login.hpc.virginia.edu:/sfs/weka/scratch/cbv8vn/TML_class/
```

The first copy may run while local work continues. Before the final copy and HPC launch, stop the local writer and repeat the sync so checkpoint files are consistent. Do not start two writers or use rsync --delete.

## HPC terminal

```bash
ssh cbv8vn@login.hpc.virginia.edu
cd /sfs/weka/scratch/cbv8vn/TML_class
module spider pytorch
module spider python
```

Select modules supported by your allocation rather than guessing version names. Prepare `.venv-hpc` using a supported Python/CUDA PyTorch environment, and install the remaining packages in `hpc/requirements.txt`. See `hpc/local_environment.json` for the Mac versions; serialized warning models require compatible scikit-learn/joblib. The Mac `.venv` is not portable. Do not install CPU-only PyTorch. Do not launch inference on the login node.

The scheduler is Slurm. Supply your actual allocation name when submitting:

```bash
sbatch --account=YOUR_ALLOCATION hpc/rivanna_a100.sbatch
squeue -u cbv8vn
```

The template requests one A100 in the gpu partition and automatically activates `.venv-hpc`. Check the GPU resource name available to your allocation; a site-specific resource can be overridden at submission. The job launcher refuses to collect on CPU and verifies the locked protocol. See UVA's GPU notes: https://learning.rc.virginia.edu/notes/llms-hpc/hpc-resources-llms/rivanna-gpu/.

Progress is under `results/final_cifar10c/hpc_run.log` and per-classifier `run.log` files. Saved weights, thresholds and image checkpoints are retained. The run remains a fresh-image confirmation under known blur, not an untouched-corruption experiment.

## Retrieve after the HPC writer has stopped

Run on the Mac from the local project root:

```bash
rsync -avh --progress cbv8vn@login.hpc.virginia.edu:/sfs/weka/scratch/cbv8vn/TML_class/results/final_cifar10c/ results/final_cifar10c/
```

No remote transfer or job submission was performed by the assistant while preparing this guide. The local collection was not stopped.

## Environment installation completed

An isolated `.python-hpc` Python base and `.venv-hpc` environment are now installed in the project scratch directory. Python 3.11.18, PyTorch 2.14.1+cu126 and torchvision 0.29.1+cu126 match the local framework releases with CUDA support. Project package consistency and locked protocol checks passed. Short validation job 22979088 detected an A100-SXM4-40GB and completed IG64 and Grad-CAM for ResNet, ViT and Swin, printing A100_CHECK_PASSED. The full study was not submitted during environment setup. Submit with `sbatch --account=uqgroup hpc/rivanna_a100.sbatch`.
