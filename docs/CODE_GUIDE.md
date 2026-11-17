# Read the code in this order

The image classifiers are frozen. The trained secondary model is **logistic regression, a binary classifier** that uses attribution features to warn about next-step first failure. Threshold calibration is a separate step using nonfailing calibration trajectories. There is no final linear-regression model.

## First: understand one image's path

| Order | File | What to look for |
|---|---|---|
| 1 | `src/cifar10c_loader.py` | How an original test ID selects the clean image and each official corruption block; severity 0 is clean |
| 2 | `src/classifiers.py` | Exact published checkpoint loading, differentiable preprocessing and architecture-specific attribution layers |
| 3 | `scripts/collect_attribution_metrics.py` | `compute_ig`, `compute_cam`, `asws_components`; follow a clean image through corruption, predictions and saved map metrics |
| 4 | `src/grad_cam.py` | Gradient hook, channel weighting, positive CAM and32×32 upsampling |
| 5 | `src/attribution_features.py` | Convert two map-generator rows into one observation, exclude invalid trajectories, add adjacent changes |
| 6 | `scripts/evaluate_early_warnings.py` | `prepare`, `calibrate_threshold`, `trajectory_outcomes`; first failure, eligible labels, strict alarms and lead |
| 7 | `scripts/run_expanded_cifar10c_study.py` | `recipes`, `make_model`, `fit_models`, `freeze`, `evaluate`; full fitting/calibration/test separation |

`src/integrated_gradients.py` is a retained standalone implementation. **The final collector actually uses Captum IG through compute_ig**; do not attribute its execution to the standalone file. `scripts/archive/custom_corruptions.py` contains the historical generated corruptions. `src/corruptions.py` is only a relative compatibility symlink to that archived file, retained for frozen source hashes and older imports. Skip it when learning the final pipeline: official collection uses `src/cifar10c_loader.py` and pre-generated pixels instead. `src/data_loader.py` supports the historical/general data path, while official pairing is in cifar10c_loader.py.

## Then: understand baselines and ablations

| File | Purpose |
|---|---|
| `src/benchmark_scores.py` | MSP/ODIN, ODIN temperature and bounded input perturbation |
| `src/supplementary_scores.py` | Stable deterministic entropy and Energy from logits |
| `scripts/evaluate_expanded_supplementary_baselines.py` | Score exact expanded cohorts, check original predictions, calibrate entropy/Energy on original groups, freeze before test |
| `scripts/evaluate_individual_lgd.py` | Six one-feature models per classifier, original fit/calibration groups, exact same test cohorts |
| `src/warning_features.py` | Checks that proposed attribution models do not silently consume confidence features |
| `src/data_reservation.py` | Audit prior experiment IDs and reserve disjoint original-test candidates |
| `src/paired_analysis.py` | Retained group-aware comparison utilities; the final descriptive report does not claim significance tests |

The primary interaction model contains 6 dynamic inputs plus 15 pairwise products; it does not square features. Individual ablations contain just L, G or D, without trends. No IG/Grad-CAM fusion or confidence-attribution hybrid is part of the final method.

## Finally: reproduce completed evaluations and presentation

| Entry point | Uses / produces |
|---|---|
| `scripts/run_expanded_cifar10c_study.py` | Main expanded collection, fit, freeze and evaluation; use existing root only to resume, not overwrite |
| `scripts/evaluate_expanded_supplementary_baselines.py` | Existing entropy/Energy supplement; fixedT=1 |
| `scripts/evaluate_individual_lgd.py` | Existing individual-component supplement |
| `scripts/run_corruption_generalization.py` | Fixed ResNet-50 panel; four additional official types, same image subset, no fitting |
| `scripts/build_final_results.py` | Reads completed evaluation CSVs and writes four figures/three tables; fastest way to inspect the work |
| `scripts/download_hf_classifiers.py` | Checkpoint download/provenance utility; existing checkpoints need not be downloaded again |
| `scripts/archive/validation/` | Historical completeness/convergence/stability checks and baseline-forward execution diagnostic |

Historical HPC launchers and environment notes are retained under `docs/archive/hpc_launchers/`; they are not needed for local reporting. Rebuilding figures needs no GPU. Recollecting all maps is expensive and unnecessary for reading or presenting results.

## Final scripts and historical sources

See `scripts/README.md` for the active scripts. All older entry points and validation implementations are archived, and obsolete top-level links have been removed. The corruption extension is named `run_corruption_generalization.py`; its outcomes remain exploratory.

Completed results, models and thresholds are unchanged. Historical frozen protocols describe their original source layout. Restore the original paths and bytes before a guarded historical rerun; do not replace the stored historical hashes. Original sources and relocation records are in `docs/archive/`. The current inventory in `configs/final_code_sha256.json` describes the cleaned local tree. This cleanup has not been synchronized to HPC.

For qualitative official same-image panels, use `scripts/build_official_attribution_examples.py`; its saved-map provenance is in `figures/official_examples/`. Manual-corruption example figures have been removed; use only the official examples for the final presentation.
