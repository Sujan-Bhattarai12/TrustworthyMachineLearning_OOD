# Rivanna and local versions

Remote project: `cbv8vn@login.hpc.virginia.edu:/sfs/weka/scratch/cbv8vn/TML_class/`.

Code used for completed evaluations was reconciled with local files before final cleanup. The final cleanup and report-generation code were subsequently mirrored back to the same remote project, with remote-only obsolete files backed up. Source hashes and exceptions are recorded in [docs/CODE_SYNC.md](../docs/CODE_SYNC.md).

Use hpc/README.md for the current launchers. The original study is completed; do not submit a new main run merely to view tables. New roots are required for changed scientific protocols. Historical source files that are hash-locked remain at original paths; never edit those to clean up naming.

To download completed final results without deleting local files:

```bash
rsync -az cbv8vn@login.hpc.virginia.edu:/sfs/weka/scratch/cbv8vn/TML_class/results/expanded_cifar10c_5000dev_2000test/ results/expanded_cifar10c_5000dev_2000test/
rsync -az cbv8vn@login.hpc.virginia.edu:/sfs/weka/scratch/cbv8vn/TML_class/results/additional_corruption_smoke/ results/additional_corruption_smoke/
```

Do not sync local virtual environments onto Linux or use --delete against datasets/results without a reviewed reason. Final code synchronization uses dedicated backup paths and a file/hash manifest; results are preserved separately.
