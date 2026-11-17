# Final rerun: resnet50

Official CIFAR-10-C subset evaluation; related custom blur was examined previously. Models and thresholds were frozen using development data only.

| Score | Role | Early / failures | False alarms | AUROC | AP |
|---|---|---:|---:|---:|---:|
| IG static | Baseline/ablation | 199/1352 | 16.32% | 0.707 | 0.435 |
| IG dynamic | Primary | 198/1352 | 15.38% | 0.709 | 0.442 |
| IG interactions | Baseline/ablation | 208/1352 | 20.08% | 0.710 | 0.424 |
| GRADCAM static | Baseline/ablation | 216/1352 | 23.45% | 0.780 | 0.505 |
| GRADCAM dynamic | Primary | 229/1352 | 22.89% | 0.781 | 0.511 |
| GRADCAM interactions | Baseline/ablation | 230/1352 | 24.39% | 0.777 | 0.522 |
| Confidence trends | Baseline/ablation | 295/1352 | 24.39% | 0.892 | 0.721 |
| MSP | Baseline/ablation | 290/1352 | 24.58% | 0.887 | 0.706 |
| ODIN | Baseline/ablation | 111/1352 | 20.45% | 0.720 | 0.461 |

A 5% development calibration target does not guarantee 5% false alarms under official Gaussian blur. All methods share the same valid cohort within this classifier; achieved false alarms differ.

Static and interaction methods are predeclared ablations; the primary IG and Grad-CAM models both use original L/G/D plus adjacent changes. Do not select a winner using these outcomes.

See comparison.csv, ranking.csv, trajectories.csv and cohort.json for exact denominators. Frozen specifications and fitted preprocessing/weights are under ../../frozen/resnet50/.
