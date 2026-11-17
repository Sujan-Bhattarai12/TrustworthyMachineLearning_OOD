> Updated protocol: use official CIFAR-10-C files and the new `results/final_cifar10c/` run. The stopped custom-corruption run is superseded; its 85 images cannot be reused as official benchmark observations. Sync the updated code, manifests and `data/CIFAR-10-C/` before submission.

# Transfer and resume on A100

The experiment is already locked. Do not run setup again, modify locked Python files, regenerate manifests or discard frozen weights. Core paths in the locked protocol are relative and portable. CUDA is selected automatically on a GPU node.

## 1. First copy from the Mac

Replace `USER`, `HPC_HOST` and the destination with your HPC details. Choose scratch/project storage with sufficient space, rather than a small home quota.

```bash
cd '/Users/sujanbhattarai/Desktop/PhD Classes/Trustworthy Machine learning/final_project'
ssh USER@HPC_HOST 'mkdir -p /path/to/project/final_project'
rsync -avh --progress --exclude='.venv/' --exclude='.git/' --exclude='__pycache__/' --exclude='.DS_Store' ./ USER@HPC_HOST:/path/to/project/final_project/
```

This initial copy may run while local collection continues. It is not the final consistent checkpoint. Do not use `--delete`, do not copy the Mac `.venv`, and do not start HPC collection yet. The project data, checkpoints and current run occupy roughly 4 GB, plus older results.

## 2. Prepare the HPC environment

Use your cluster's CUDA-enabled PyTorch environment or install a CUDA PyTorch/torchvision pair compatible with its driver. Match `hpc/local_environment.json` where possible, especially scikit-learn/joblib for serialized warning models. If exact framework versions are unavailable, record and verify compatibility rather than silently substituting versions. No half precision, compilation or algorithm changes are needed for this migration.

For example, after loading the cluster's supported Python/CUDA modules:

```bash
cd /path/to/project/final_project
python3 -m venv .venv-hpc
source .venv-hpc/bin/activate
# Install the appropriate CUDA PyTorch + torchvision using your cluster's instructions.
python -m pip install -r hpc/requirements.txt
```

CUDA availability must be checked inside an allocated GPU job, not necessarily on the login node. Already downloaded checkpoint weights are transferred; no Hugging Face downloads are needed. Pinning dependencies may require internet on a login node or your cluster's wheel mirror.

## 3. Stop local collection, then sync again

Return to the agent for a checkpoint-aware local stop before switching. Do not allow two orchestrators to write the same experiment. If manually interrupting a foreground run, use Ctrl+C; the collector checkpoints every five completed images, so a few recently processed images may be recomputed. Do not delete an incomplete run folder.

After the local writer has stopped, repeat the same `rsync` command. Ensure `results/final_cifar10c/locked_protocol.json`, manifests, reservation audit, all `development/`, `test/`, `preflight/` and any `frozen/` files are included. Historical status may still say running after an interruption; the new orchestrator updates it when resuming. Do not interpret that stale field alone as a live process.

## 4. Validate and run on an allocated GPU

```bash
cd /path/to/project/final_project
source .venv-hpc/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -c "from pathlib import Path; from scripts.run_cifar10c_study import check; check(Path('results/final_cifar10c')); print('Locked code/manifests/provenance verified')"
bash hpc/run_on_gpu.sh
```

The launcher requires CUDA, checks locked code/manifests, records the new software/GPU environment, then resumes with a fresh authorized 10-hour invocation. Framework loading verifies weight hashes. Match a few cached image predictions/maps across CPU and CUDA before interpreting resumed numerical results; an A100 can produce small floating-point differences. Preserving completed CPU data gives a documented mixed-device run, not a from-scratch GPU run.

For **Slurm only**, `hpc/slurm_a100.sbatch` is a template. Set the correct partition/account and GPU syntax for your cluster, activate the environment in the submission shell and submit from the project root:

```bash
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT hpc/slurm_a100.sbatch
```

If your scheduler is not Slurm, use the same GPU launcher inside that scheduler's allocation. Never launch the heavy run on a login node. No submission or remote transfer has been performed by the agent.

## 5. Follow progress and retrieve results

```bash
tail -f results/final_cifar10c/hpc_run.log
tail -f results/final_cifar10c/development/resnet50/run.log
```

After the HPC job is stopped/finished, copy results back from the Mac:

```bash
rsync -avh --progress USER@HPC_HOST:/path/to/project/final_project/results/final_cifar10c/ results/final_cifar10c/
```

Do not sync local run files over an actively writing HPC copy. No GPU-runtime promise is made before a real CUDA throughput benchmark.
