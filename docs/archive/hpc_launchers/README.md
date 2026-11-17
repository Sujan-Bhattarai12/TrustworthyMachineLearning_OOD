# Rivanna execution

Completed final study directory: `/sfs/weka/scratch/cbv8vn/TML_class`.

| Launcher | Purpose |
|---|---|
| expanded_cifar10c_a100.sbatch | Three-architecture expanded collection, fit/freeze/test; A100,48h limit |
| expanded_supplementary_baselines.sbatch | Entropy/Energy on the exact expanded cohorts, after primary evaluation |
| individual_lgd_ablations.sbatch | CPU-only individual L/G/D ablations on original frozen groups |
| additional_corruptions.sbatch | ResNet-only frozen generalization panel |
| create_environment.sh | Environment setup; existing environment need not be recreated |
| validation/test_a100.sbatch | Historical GPU attribution execution check |

Final supplementary launch order is main first, then entropy/Energy and individual ablations; panel requires those frozen artifacts. All jobs have completed. Rebuilding figures is local CPU work and requires no new SLURM job. See [reproducibility/sync notes](../docs/CODE_SYNC.md), [study specification](../docs/FINAL_STUDY.md) and [code reading guide](../docs/CODE_GUIDE.md).

`archive/previous_runs/` preserves old500-image and failed supplementary launchers; these are not the final execution path. Environment records installed_environment.json, local_environment.json and per-run hpc_environment files record actual versions. Use requirements.txt/requirements-hf.txt together with the recorded PyTorch/CUDA platform; a package name alone does not reproduce GPU execution.
