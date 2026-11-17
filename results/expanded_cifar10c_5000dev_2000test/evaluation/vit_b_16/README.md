# Final rerun: vit_b_16

Official CIFAR-10-C subset evaluation; related custom blur was examined previously. Models and thresholds were frozen using development data only.

| Score | Role | Early / failures | False alarms | AUROC | AP |
|---|---|---:|---:|---:|---:|
| IG static | Baseline/ablation | 6/110 | 5.50% | 0.755 | 0.084 |
| IG dynamic | Primary | 12/110 | 5.71% | 0.761 | 0.097 |
| IG interactions | Baseline/ablation | 2/110 | 4.15% | 0.737 | 0.068 |
| GRADCAM static | Baseline/ablation | 5/110 | 3.94% | 0.770 | 0.065 |
| GRADCAM dynamic | Primary | 6/110 | 3.99% | 0.786 | 0.081 |
| GRADCAM interactions | Baseline/ablation | 9/110 | 4.31% | 0.812 | 0.097 |
| Confidence trends | Baseline/ablation | 32/110 | 2.43% | 0.982 | 0.560 |
| MSP | Baseline/ablation | 30/110 | 2.37% | 0.985 | 0.616 |
| ODIN | Baseline/ablation | 1/110 | 0.05% | 0.860 | 0.168 |

A 5% development calibration target does not guarantee 5% false alarms under official Gaussian blur. All methods share the same valid cohort within this classifier; achieved false alarms differ.

Static and interaction methods are predeclared ablations; the primary IG and Grad-CAM models both use original L/G/D plus adjacent changes. Do not select a winner using these outcomes.

See comparison.csv, ranking.csv, trajectories.csv and cohort.json for exact denominators. Frozen specifications and fitted preprocessing/weights are under ../../frozen/vit_b_16/.
