# Final study scripts

The retained final-study and reporting entry points are:

| Script | Purpose |
|---|---|
| `run_expanded_cifar10c_study.py` | Main 5,000-development / 2,000-test candidate study, warning fitting, freezing and evaluation |
| `collect_attribution_metrics.py` | Predictions, IG/Grad-CAM maps and attribution measurements |
| `evaluate_early_warnings.py` | Shared next-step labels, threshold calibration and trajectory outcomes |
| `evaluate_expanded_supplementary_baselines.py` | Entropy and Energy on final cohorts |
| `evaluate_individual_lgd.py` | Individual L, G and D ablations |
| `run_corruption_generalization.py` | Exploratory frozen ResNet panel on four additional official corruptions |
| `build_official_attribution_examples.py` | Same-image official severity and saved attribution panels |
| `build_final_results.py` | Final figures and tables from completed observations |
| `download_hf_classifiers.py` | Download pinned checkpoint files and record provenance |

## Archive

Historical runners, Week 4 builders, custom experiments and their helper implementations are under `archive/`. Stability/completeness/diagnostic scripts are under `archive/validation/`. Root compatibility links to obsolete scripts have been removed. Archived code is research history, not the final workflow.

## Reproducibility

Completed observations, maps, fitted models, thresholds, result tables and frozen protocol JSONs were not changed. `configs/final_code_sha256.json` inventories the current source tree; historical protocols continue to describe the original execution sources. Originals and relocation records are under `docs/archive/`. Source guards intentionally reject altered or missing historical paths: restore historical bytes and paths before resuming a locked historical run. Do not rewrite its hashes to bypass the checks. Current reporting uses completed CSVs and requires no GPU/refit.

The collector retains historical custom-data defaults; the final runner passes `--dataset cifar10c` and explicit manifests. The retained four-corruption extension is exploratory, not a new confirmatory experiment.

This cleanup is local; Rivanna has not been synchronized.
