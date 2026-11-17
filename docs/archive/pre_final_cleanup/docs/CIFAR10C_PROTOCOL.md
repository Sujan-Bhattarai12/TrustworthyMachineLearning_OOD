# Final study: official CIFAR-10-C subset

This protocol supersedes the stopped custom-corruption run. Use the pre-generated `.npy` images in `data/CIFAR-10-C/`. Do not regenerate, interpolate or replace official pixels with `src/corruptions.py` outputs.

Source: https://zenodo.org/records/2535967 and https://github.com/hendrycks/robustness. The official corruption images are synthetic; the advantage is benchmark standardization and comparability, not inherently more reliable warning performance.

## Pairing and severity

For original CIFAR-10 test ID i, corruption severity s in 1–5 is row `(s-1)*10000+i` of the chosen corruption file. Severity 0 is the original clean CIFAR-10 test image i. `labels.npy` repeats the test label order across all five 10,000-image blocks. The loader checks labels, shapes and uint8 format, reads memory-mapped arrays, transposes HWC to CHW and scales by 255 before checkpoint preprocessing. No custom corruption is applied.

## Disjoint subsets

Official CIFAR-10-C contains corruptions of the original 10,000 **test** images, not training images. The old training-source development manifest therefore cannot be used. A 1,200-image pool is reserved outside recorded prior warning-experiment candidate IDs, label-stratified with seed 20261114. Within each class, shuffle and assign 100 images to development candidates and 20 to final evaluation. This gives 1,000 development candidates and 200 final candidates, with no ID overlap.

Per classifier, select 50 clean-correct development images per class (500 total) using clean predictions only. Split common-valid development image identities into 75% fitting and 25% calibration, random seed42. All severity levels and both development corruptions stay with their image. Final evaluation processes all clean-correct images among the 200 candidates and reports actual exclusions/counts.

This is a **supervised warning-detector study on a CIFAR-10-C subset**. It uses a development subset of benchmark test-source images for the secondary warning model; it is not an unmodified 10,000-image/19-corruption benchmark score. The frozen classifiers are not trained or fine-tuned here. Test-source image exposure in external checkpoint training is not independently established.

## Fixed final configuration

- Frozen ResNet-50, ViT-B/16 and Swin-T checkpoints and their original preprocessing.
- Development: official `gaussian_noise.npy` and `brightness.npy`.
- Final corruption: official `gaussian_blur.npy` (one of the distributed corruption files).
- Severity 0–5, IG64, positive Grad-CAM; fixed clean predicted class target.
- Original L/G/D definitions and shared-clean-maximum map normalization.
- Primary: separate IG and Grad-CAM logistic models with L/G/D and adjacent-level differences.
- Ablations: static L/G/D logistic models and pairwise interaction models.
- Baselines: MSP, ODIN and a separate confidence-trend logistic model.
- No confidence in attribution predictors. Calibrate thresholds on nonfailing development-calibration trajectories at a nominal 5% trajectory false-alarm target. Save fitted weights/scalers and thresholds before final predictions are scored.
- Report all predeclared methods, early/missed warnings, lead in severity steps, achieved false alarms and secondary AUROC/AP. No final-outcome method selection or threshold retuning.

The five levels are static benchmark severities, not real time. Random corruption realizations may differ between levels, unlike our earlier fixed-direction custom noise. A first observed failure is a failure on this sampled severity sequence, not an exact continuous boundary.

## Provenance

The local dataset directory contains all 19 corruption files. Selected file shapes and all five label blocks match the official format. SHA-256 hashes of the used files are recorded in `results/final_cifar10c/locked_protocol.json`. The original download tar is absent, so the published archive MD5 has **not** been independently verified. Local hashes prevent drift but are not proof of byte-level identity to a separately verified release archive.

Related custom Gaussian blur was previously examined. Fresh image IDs and a switch to official files do not erase prior research exposure. Describe Gaussian blur as held out from this warning-model fit, not proof of broad untouched OOD generalization.

## Execution

Setup is complete. Do not rerun setup or resume the old custom run. Sync this project to Rivanna again, including `data/CIFAR-10-C/`, `results/final_cifar10c/`, code and checkpoints. `hpc/run_on_gpu.sh` now verifies and runs:

```bash
python -m scripts.run_cifar10c_study run --hours 10
```

The local run remains stopped. The A100 launcher checks CUDA and preserves resumable progress. No official-image attribution collection has been launched during protocol conversion.

## Historical evidence

All previous figures/tables and `docs/WEEK4_EVALUATION.md` describe custom-corruption experiments unless explicitly stated otherwise. They must not be relabeled as CIFAR-10-C results. The 85 checkpointed custom images in `results/final_project_run/` are superseded and not mixed with this run.
