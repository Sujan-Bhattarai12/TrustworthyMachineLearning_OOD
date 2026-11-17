> **Scope update:** The new final study uses [official CIFAR-10-C files](CIFAR10C_PROTOCOL.md). The figures and results below remain historical custom-corruption evidence until replacement outputs exist. Do not relabel them.

# Presentation guide

Use existing evidence. No new experiments are required for presentation preparation.

## Suggested narrative and materials

1. **Question and pipeline:** attribution-only early warnings versus confidence, using frozen ResNet-50, ViT-B/16 and Swin-T. Explain IG/Grad-CAM → L/G/D → warning score → frozen threshold. Distinguish learned logistic models from fixed selected blur scores.
2. **Corruption and attribution examples:** [severity maps](../figures/examples/asws_severity_maps.png), [confidence example](../figures/examples/asws_severity_confidence.png), [corruption examples](../figures/examples/development_corruptions.png). These are historical illustrations; do not label them as a new 64-step run or assume every example uses the final classifiers without checking its provenance.
3. **Attribution validation:** [IG stability](../figures/validation/week1_dev_ig_stability.png), [IG convergence heatmaps](../figures/validation/week1_dev_ig_convergence_heatmaps.png), [Grad-CAM stability](../figures/validation/week1_dev_gradcam_stability.png), [Grad-CAM heatmaps](../figures/validation/week1_dev_gradcam_heatmaps.png). Supporting CSVs: `results/validation/`. These are earlier WideResNet/RobustBench validation checks, not proof of completeness or stability for every final checkpoint. IG convergence compares attribution estimates and completeness residuals; Grad-CAM has no corresponding IG completeness axiom.
4. **Attribution/error association:** [figure guide and table](../results/reporting/attribution_error_relationships/README.md). Start with [ResNet noise](../results/reporting/attribution_error_relationships/resnet50_gaussian_noise.pdf) and [ResNet brightness](../results/reporting/attribution_error_relationships/resnet50_brightness.pdf). All three classifiers have IG/Grad-CAM L/G/D median/IQR curves with correct/incorrect counts. These describe current errors, not pre-failure warning evidence. D includes retention; declining D can accompany attribution collapse.
5. **Frozen evaluation:** [Week 4 report](WEEK4_EVALUATION.md), [timing/false alarms](../results/reporting/week4/timing_and_false_alarms.pdf), [ROC/PR](../results/reporting/week4/roc_pr.pdf). CSV tables: `results/reporting/week4/{timing,ranking,cohorts,frozen_specifications}.csv`. Keep classifier-specific cohorts and score definitions explicit.
6. **Conclusions and limitations:** attribution contains predictive signal, but frozen static blur scores did not consistently improve over MSP. A 5% calibration target did not guarantee 5% evaluation false alarms. Development exposure, clean-reference requirements and severity-based (not real-time) lead are limitations. Later dynamics/normalization results are exploratory, not replacements for the original frozen evaluation.

## Code to explain

- `src/data_loader.py`: original CIFAR-10 and manifest loading.
- `src/classifiers.py`: frozen checkpoints, preprocessing and attribution layers.
- `src/corruptions.py`: primary discrete severity levels 0–5; optional historical continuous helper is not the presentation protocol.
- `src/integrated_gradients.py`, `src/grad_cam.py`: attribution implementations. Main collection uses Captum IG in `scripts/collect_attribution_metrics.py`.
- `scripts/collect_attribution_metrics.py`: L/G/D definitions, map collection and step configuration.
- `src/attribution_features.py`: reusable feature/trend preparation.
- `src/warning_features.py`: attribution-only input guard.
- `scripts/evaluate_warning_models.py`, `scripts/evaluate_early_warnings.py`: learned scores and timing evaluation.
- `scripts/freeze_warning_methods.py`, `scripts/evaluate_frozen_warnings.py`: selected frozen score artifacts and evaluation.
- `scripts/plot_attribution_error_relationships.py`, `scripts/build_week4_report.py`: final figures/tables from saved observations.

## IG configuration

New integration counts: powers of two 1–128; default 64. The completed main evidence used 50 steps and remains labeled accordingly. Changing a source-code default does not retrospectively change results. Validation reruns use the new convention; no reruns were performed in cleanup.

## Archive policy

Exploratory scripts/results remain archived, not deleted or promoted as final evidence. Historical documents and locked JSON retain original path text; use the [archive index](ARCHIVE_INDEX.md) and relocation manifests to resolve it. No original test images, manifests or checkpoint weights were changed. Archived CSVs remain recursively visible to the historical reservation checks.
