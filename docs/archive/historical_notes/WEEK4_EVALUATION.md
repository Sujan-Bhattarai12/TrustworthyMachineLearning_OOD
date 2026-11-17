# Week 4: completed warning-method evaluation

## Main finding

The completed frozen blur evaluation does not demonstrate consistent attribution superiority over MSP. The selected attribution score detects fewer failures early than MSP on all three classifiers. Development calibration targeting 5% trajectory false alarms does not reliably transfer to blur. These are completed evaluation outcomes, not new experiments or a newly selected winner.

## What was actually evaluated

This primary table uses only the original saved frozen attribution-only specifications and their exact blur cohorts. ResNet and ViT use fixed Grad-CAM L+D; Swin uses raw Grad-CAM L. These are not the later learned trend, fusion, interaction, DRO or V-REx scores. A logistic warning framework exists, but it would be inaccurate to describe these particular final results as logistic models.

Development fitting/scaling and threshold calibration used disjoint training-source images under noise and brightness. Attribution combinations were selected from completed development summaries by credited lead subject to a development false-alarm budget. Selection details and coefficients/scales are retained in the frozen artifacts. Classifiers were not retrained. MSP and ODIN were calibrated separately with the same policy. The original frozen suite did not include confidence trends; its absence here does not imply it was evaluated or defeated.

## Frozen thresholds

| classifier | model | threshold | calibration_negatives | calibration_fpr |
| --- | --- | --- | --- | --- |
| resnet50 | MSP | 0.166891 | 123 | 4.88% |
| resnet50 | ODIN | 0.899128 | 123 | 4.88% |
| resnet50 | GRADCAM L+D fixed | 1.111163 | 123 | 4.88% |
| vit_b_16 | MSP | 0.354603 | 172 | 4.65% |
| vit_b_16 | ODIN | 0.899626 | 172 | 4.65% |
| vit_b_16 | GRADCAM L+D fixed | 0.800832 | 172 | 4.65% |
| swin_t | MSP | 0.520869 | 137 | 4.38% |
| swin_t | ODIN | 0.899659 | 137 | 4.38% |
| swin_t | GRADCAM L raw | 0.561429 | 137 | 4.38% |

Fixed L+D averages L and D after dividing by fitting-only 95th-percentile scales; raw L uses no such averaging. These scales are normalization choices, not probability thresholds. Exact scales and input columns are in `frozen_specifications.csv`.

## Cohorts

| Classifier | candidates | clean_correct | valid_trajectories | excluded_trajectories | failures | nonfailures | fit_images | calibration_images |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet50 | 500 | 472 | 472 | 0 | 404 | 68 | 375 | 125 |
| vit_b_16 | 500 | 497 | 497 | 0 | 134 | 363 | 375 | 125 |
| swin_t | 500 | 486 | 480 | 6 | 133 | 347 | 372 | 125 |

Cohorts are shared across methods within each classifier, but differ across classifiers. Clean-incorrect candidates and invalid attribution trajectories are excluded; results are conditional on this selection. Failure means at least one incorrect prediction by blur severity 5, even if the classifier later recovers.

## Primary timing and secondary ranking results

| Classifier | Score | Early / failures | Early % | False alarm % | Mean lead | AUROC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 | MSP | 87 / 404 | 21.5 | 39.7 | 0.230 | 0.718 | 0.605 |
| ResNet-50 | ODIN | 41 / 404 | 10.1 | 27.9 | 0.119 | 0.632 | 0.529 |
| ResNet-50 | GRADCAM L+D fixed | 48 / 404 | 11.9 | 35.3 | 0.146 | 0.643 | 0.502 |
| ViT-B/16 | MSP | 18 / 134 | 13.4 | 8.5 | 0.134 | 0.946 | 0.566 |
| ViT-B/16 | ODIN | 0 / 134 | 0.0 | 0.0 | 0.000 | 0.829 | 0.282 |
| ViT-B/16 | GRADCAM L+D fixed | 13 / 134 | 9.7 | 14.3 | 0.157 | 0.686 | 0.128 |
| Swin-T | MSP | 17 / 133 | 12.8 | 8.1 | 0.128 | 0.940 | 0.573 |
| Swin-T | ODIN | 0 / 133 | 0.0 | 0.0 | 0.000 | 0.853 | 0.309 |
| Swin-T | GRADCAM L raw | 10 / 133 | 7.5 | 23.9 | 0.120 | 0.588 | 0.097 |

Early warnings occur strictly before first failure. Mean lead is measured in severity steps and includes missed failures as zero. False-alarm denominators are nonfailing trajectories, not all images. AUROC and AP use still-correct, nonzero-severity observations with an observable next step (levels 1–4 before first failure); they assess a different task from trajectory timing. First-level failures cannot receive a preceding nonzero-level warning. AP is average precision, not trapezoidal PR area. Secondary ranking was computed retrospectively from frozen scores without fitting or calibration.

![Timing and false alarms](../../results/reporting/week4/timing_and_false_alarms.png)

The dashed line is a development calibration target, not guaranteed test control. Higher early-warning counts must be interpreted alongside achieved false alarms. No test thresholds were adjusted to force equal false-alarm rates.

![ROC and precision–recall](../../results/reporting/week4/roc_pr.png)

PR prevalence and eligible observation counts differ across classifiers. Do not compare AP across cohorts as if the tasks were identical. Curves and bars are descriptive point estimates without confidence intervals; they do not establish statistical significance.

## Secondary ranking denominators

| classifier | n_images | n_steps | n_next_step_failures | failure_prevalence |
| --- | --- | --- | --- | --- |
| resnet50 | 460 | 1023 | 392 | 38.32% |
| vit_b_16 | 494 | 1875 | 131 | 6.99% |
| swin_t | 475 | 1758 | 128 | 7.28% |

## Supporting evidence, kept separate

- [Feature diagnosis](../../results/archive/exploratory/attribution_feature_diagnosis/FINDINGS.md): L/G/D contain heterogeneous signals depending on classifier, corruption and comparison population.
- [Interaction ablation](../../results/archive/exploratory/resnet_attribution_interactions_v2/FINDINGS.md): Grad-CAM with L/G/D trends and interactions improved mixed-corruption ResNet development detection (161/515 versus 137 additive), with 5.57% versus 5.98% false alarms. This is exploratory reused-development evidence.
- [Corruption transfer](../../results/archive/exploratory/resnet_corruption_transfer/FINDINGS.md): the interaction gains did not reliably transfer. Noise→brightness detected 4/56 early with 4.50% false alarms; brightness→noise detected 90/459 with 80.49% false alarms.
- DRO/V-REx, historical hybrids and early pilots are [archived](../../results/README.md). They are not pooled into the main comparison or used to claim attribution-only superiority.

## Limitations and report conclusion

Blur was originally held out but its outcomes have since been inspected during the broader research process. It is now a previously examined condition, not an untouched corruption for future tuning. Published classifiers may have seen development images during original training; external training exposure is not independently established. Maps require a clean reference and a fixed clean predicted class; trends require ordered observations. Severity steps are not real time, so lead time is not seconds. Small nonfailing groups make false-alarm estimates sensitive to individual images.

The defensible conclusion is that attribution changes carry useful predictive information, while the present warning scores do not establish consistent superiority over confidence or robust false-alarm control under corruption shift. Week 4 evaluation is consolidated; report discussion and failure analysis remain. Method expansion remains closed.

## Reproducibility

[CSV tables, curves, figures and source hashes](../../results/reporting/week4) are generated by `scripts/build_week4_report.py`. It verifies frozen artifacts against those used for evaluation, development hashes and reconstructed warning counts before rendering. No source result file is overwritten, and no collection, fitting or threshold optimization is performed. Generated output directory must be absent to rebuild; preserve it before regenerating.
