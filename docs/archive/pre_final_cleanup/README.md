> **Protocol update:** The final run now uses official CIFAR-10-C files. See [official protocol](docs/CIFAR10C_PROTOCOL.md). Previous custom-corruption results remain historical and must not be presented as CIFAR-10-C benchmark results. New run: `results/final_cifar10c/`; the stopped custom run is superseded.

# Attribution-Based Failure Detection Under Distribution Shift

Does attribution change warn of classification failure earlier or more reliably than model confidence? This repository contains the completed CIFAR-10 study and presentation materials. Method expansion is closed.

**Start with the [presentation guide](docs/PRESENTATION_GUIDE.md).** It identifies the figures, tables, validation evidence and exact experiments to present.

## Retained pipeline

1. Load frozen CIFAR-10 **ResNet-50, ViT-B/16 or Swin-T** checkpoints.
2. Apply **Gaussian noise, brightness or blur at severity 0–5**, where 0 is clean.
3. Generate IG and Grad-CAM maps for the clean predicted class.
4. Compute custom attribution loss L, gain G and retention-weighted Wasserstein distance D.
5. Fit/calibrate attribution-only warning scores and compare with **MSP, ODIN and separate confidence trends**.
6. Evaluate early warnings, missed warnings, lead, achieved false alarms, AUROC and AP.

Confidence, labels and severity are not inputs to proposed attribution scores. IG/Grad-CAM are map generators, not warning baselines. Development uses noise and brightness; blur was originally held out and has now been examined. Original data, checkpoints and split manifests are preserved.

## Current settings versus historical evidence

New IG runs default to **64 integration steps**. Allowed values are **1, 2, 4, 8, 16, 32, 64 and 128** (`--ig-steps`). Use 32 for fewer steps than the previous 50. Powers of two are a configuration convention, not a RAM requirement. Existing **50-step results remain historical 50-step results**, with explicit replay support for frozen protocols. They were not regenerated or relabeled during cleanup.

The original frozen blur scores were development-selected fixed Grad-CAM L+D for ResNet/ViT and raw L for Swin. Later learned logistic/trend/interactions are distinct experiments. Attribution signals were useful in development, but consistent superiority over confidence and reliable false-alarm control under shift were not established.

## Where things live

| Location | Contents |
|---|---|
| `src/` | Core data, classifiers, corruptions, attribution methods/features and paired analysis |
| `scripts/` | Core collection, fitting, frozen evaluation and final-report plotting |
| `scripts/validation/` | IG completeness/convergence and Grad-CAM agreement/stability checks |
| `results/classifier_benchmarks_500/` | Original development observations for all three classifiers |
| `results/frozen_warning_methods/`, `results/final_blur_evaluation/` | Original frozen scores and evaluation |
| `results/reporting/` | Presentation figures, tables and source hashes |
| `results/validation/`, `figures/validation/` | Preserved historical attribution-validation evidence |
| `figures/examples/` | Existing corruption/image/attribution illustrations |
| `scripts/archive/`, `tests/archive/`, `results/archive/`, `docs/archive/` | Exploratory/superseded work and reproducibility history |
| `data/`, `splits/`, `checkpoints/`, `configs/` | Original assets and provenance; unchanged |

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/asws-mpl .venv/bin/python -m unittest discover -s tests -q
```

The full suite includes archived regression coverage. No collectors, checkpoints or datasets need to be rerun/downloaded to view the report. Dependencies are recorded in `requirements-hf.txt`.

See the [Week 4 report](docs/WEEK4_EVALUATION.md), [completion plan](PROJECT_PLAN.md), [results guide](results/README.md) and [archive index](docs/ARCHIVE_INDEX.md).

## Final rerun

The authorized [final project run](results/final_project_run/README.md) uses 500 development images and 200 fresh test candidates per classifier, IG64 and primary dynamic attribution-only logistic scores. See [locked protocol](results/final_project_run/locked_protocol.json) and [progress log](results/final_project_run/run.log). Historical results are not overwritten.

Supplementary comparisons now include deterministic predictive entropy and post-hoc Energy. See [baseline formulas, paper citations and protocol](docs/SUPPLEMENTARY_BASELINES.md). These are separate additions to the locked primary CIFAR-10-C run and require only forward inference.

**Current method:** original attribution-only L/G/D pipeline (IG and Grad-CAM; existing trend features retained). Pearson/cosine, sparse encodings, dot-product/projection variants and their combinations are discontinued. Their [exploratory evidence is archived](results/archive/map_similarity/README.md); they are not part of the final method.

The expanded study also includes [individual L, G and D ablations](docs/INDIVIDUAL_LGD_ABLATIONS.md) for both IG and Grad-CAM, using the same fitting/calibration groups and test cohorts as the combined scores.
