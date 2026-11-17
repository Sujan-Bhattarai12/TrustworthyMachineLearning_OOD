# Final rerun: swin_t

Official CIFAR-10-C subset evaluation; related custom blur was examined previously. Models and thresholds were frozen using development data only.

| Score | Role | Early / failures | False alarms | AUROC | AP |
|---|---|---:|---:|---:|---:|
| IG static | Baseline/ablation | 5/211 | 8.19% | 0.553 | 0.037 |
| IG dynamic | Primary | 5/211 | 9.01% | 0.559 | 0.039 |
| IG interactions | Baseline/ablation | 1/211 | 6.32% | 0.551 | 0.035 |
| GRADCAM static | Baseline/ablation | 5/211 | 4.27% | 0.578 | 0.038 |
| GRADCAM dynamic | Primary | 3/211 | 4.21% | 0.579 | 0.038 |
| GRADCAM interactions | Baseline/ablation | 5/211 | 7.31% | 0.580 | 0.036 |
| Confidence trends | Baseline/ablation | 42/211 | 2.28% | 0.968 | 0.540 |
| MSP | Baseline/ablation | 52/211 | 2.63% | 0.966 | 0.499 |
| ODIN | Baseline/ablation | 3/211 | 0.00% | 0.880 | 0.198 |

A 5% development calibration target does not guarantee 5% false alarms under official Gaussian blur. All methods share the same valid cohort within this classifier; achieved false alarms differ.

Static and interaction methods are predeclared ablations; the primary IG and Grad-CAM models both use original L/G/D plus adjacent changes. Do not select a winner using these outcomes.

See comparison.csv, ranking.csv, trajectories.csv and cohort.json for exact denominators. Frozen specifications and fitted preprocessing/weights are under ../../frozen/swin_t/.
