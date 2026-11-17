# Historical experiment log — includes confidence–attribution hybrids

These experiments are preserved for provenance. Hybrid scores do not test standalone attribution superiority and are excluded from the corrected primary method.

# Attribution-Based Early Warning of Classification Failure

## Project goal

Evaluate whether attribution loss (L), attribution gain (G), and retention-weighted Wasserstein distance (D) warn of classification failure before model confidence, at comparable false-alarm rates as corruption increases. IG and Grad-CAM generate the maps; they are not competing warning baselines.

## Published CIFAR-10 classifiers

No additional fine-tuning is required for these already trained CIFAR-10 checkpoints:

| Classifier | Published checkpoint | Input preprocessing |
|---|---|---|
| ResNet-50 | [edadaltocg/resnet50_cifar10](https://huggingface.co/edadaltocg/resnet50_cifar10) | 32×32; CIFAR mean/std; small 3×3 stem, no initial max-pool |
| ViT-B/16 | [jialicheng/cifar10_vit-base](https://huggingface.co/jialicheng/cifar10_vit-base) | 224×224 bilinear resize; mean/std 0.5 |
| Swin-T | [jcollado/swin-tiny-patch4-window7-224-finetuned-cifar10](https://huggingface.co/jcollado/swin-tiny-patch4-window7-224-finetuned-cifar10) | 224×224 bicubic resize; ImageNet mean/std |

Pinned repository revisions are in `configs/hf_classifiers.json`. Downloads record file SHA-256 hashes and provenance under `checkpoints/huggingface/<classifier>/`. Model weights load strictly, without executing repository Python code. ViT/Swin label metadata must exactly match CIFAR-10; ResNet uses the repository's standard CIFAR-10 dataset order, also checked with development predictions.

The previously suggested `nielsr/swin-tiny-patch4-window7-224-finetuned-cifar10` is not used: its configuration has EuroSAT labels despite the repository name.

Preprocessing is differentiable float-tensor resizing/normalization so ODIN and attribution gradients reach the original [0,1] pixels. It avoids PIL quantization; tiny differences from a PIL-based reference pipeline can occur.

## Benchmark baselines

- **MSP:** `1 - max(softmax(logits))`. [Hendrycks and Gimpel](https://arxiv.org/abs/1610.02136)
- **ODIN:** temperature-scaled confidence after a confidence-increasing input step. Defaults: temperature 1000, epsilon 0.0014 in original [0,1] pixel units; the perturbed image is clipped. This bounded-pixel adaptation differentiates through resizing/normalization. Parameters require development-only tuning. [Liang et al.](https://arxiv.org/abs/1706.02690)


## Planned attribution-score ablations

For each classifier, evaluate these scores separately using IG-derived maps and Grad-CAM-derived maps:

| Score | Question |
|---|---|
| L only | Does attribution loss alone predict upcoming failure? |
| G only | Does attribution gain alone predict upcoming failure? |
| D only | Does retention-weighted Wasserstein distance alone predict upcoming failure? |
| L + G, L + D, G + D | Which pairs provide complementary warning information? |
| L + G + D | Does the full combination improve over individual components and pairs? |

Evaluate raw L, G, and D directly, in addition to single-component logistic-regression variants. For fixed combined scores, scale components using only the fitting portion of development data and average the selected components with equal weights. Compare these fixed scores with learned logistic combinations, labeling the two approaches separately. The primary timing evaluator implements raw individual scores, fixed pairs/full combinations, and learned full L/G/D scores. Fixed combinations divide each component by its fitting-only 95th percentile (fallback 1 for a zero percentile) and average the selected components. The secondary ranking evaluator also includes learned individual-component variants.

Use the same eligible examples and image-grouped splits for every ablation and the MSP/ODIN baselines. Report next-severity AUROC/AUPR and, after development-only threshold calibration, warning lead time at comparable false-alarm rates. Select the final score using development results and freeze its components, scaling, weights, and threshold before final testing. An individual component may be selected if it performs better than the combinations.

## Primary endpoint: warning before first failure

The main question is **whether attribution warns earlier than MSP/ODIN**, rather than whether it has higher AUROC/AUPR.

`python -m scripts.fit_aws_logreg --input PATH --output-dir OUTPUT` now runs timing evaluation first. For a standalone timing evaluation:

```bash
python -m scripts.evaluate_early_warnings --input results/classifier_benchmarks/resnet50/images.csv --output-dir results/classifier_benchmarks/resnet50/comparison/early_warning --target-fpr 0.05
```

- Five outer image-grouped folds evaluate unseen warning-evaluation images. Within each outer training partition, 25% of image groups calibrate thresholds and the rest fit score scales/weights. Both corruption trajectories of an image always stay together.
- Monitor nonzero severities 1–5. First failure is the first incorrect prediction, even if the classifier later recovers. Only alarms strictly before first failure count as early warnings. Failure at severity 1 has no monitored pre-failure opportunity and remains a missed warning.
- Calibrate each method with the same 5% empirical **trajectory-level** false-alarm target: any alarm over severities 1–5 on a calibration trajectory that never fails. A strict `score > threshold` rule handles ties conservatively. Calibration uses one pooled threshold per method/fold across both development corruptions. Small negative calibration samples may only permit zero empirical false alarms. If no negative calibration trajectories exist, skip and report that fold rather than using evaluation data.
- Report early-warning rate, missed-warning counts, average credited lead over **all failures** (zero for misses), conditional lead among detected failures, and achieved evaluation trajectory false-alarm rate, separately per corruption and pooled. Lead is in severity steps, not seconds.
- Paired comparisons against MSP and ODIN use shared failing trajectories. Zero-credit ties can include both methods missing the failure.
- A common calibration target does **not** guarantee equal evaluation false-alarm rates. Inspect achieved rates before making a matched-false-alarm superiority claim. No evaluation-label retuning is allowed.
- Invalid/missing attribution at any severity excludes the entire trajectory for all methods and is reported. Complete severities 0–5 are required, so partial collection checkpoints are not silently interpreted as nonfailure trajectories.

Primary outputs: `comparison/summary.csv` and `comparison/early_warning/{summary,trajectories,thresholds,splits,paired_comparisons,excluded_trajectories}.csv`, plus `protocol.json`. Secondary AUROC/AUPR outputs: `comparison/secondary_next_step_summary.csv` and `secondary_next_step_predictions.csv`. Historical `predictions.csv` files may still exist from the previous evaluator and should not be mistaken for current outputs.

This remains development cross-validation, not the reserved blur/test evaluation. Freeze the final method and calibration protocol before final testing.

## Run

Install the checkpoint-loading dependency in the existing project environment:

```bash
python -m pip install -r requirements-hf.txt
python -m scripts.download_hf_classifiers
python -m scripts.run_classifier_benchmarks --n-per-class 5
```

The launcher runs the three classifiers sequentially and saves collection/comparison outputs separately under `results/classifier_benchmarks/<classifier>/`. Five images per class means 50 images for each classifier. Use 50 per class for a 500-image run after validation.

For a one-image end-to-end smoke run:

```bash
python -m scripts.run_asws_experiment --classifier vit_b_16 --checkpoint checkpoints/huggingface/vit_b_16 --n-per-class 1 --max-images 1 --output-dir results/published_smoke/vit_b_16
```

A single image cannot support image-grouped cross-validation. Smoke outputs are implementation checks, not benchmark performance results.

The original local checkpoint loader remains available for explicitly supplied compatible `.pt` files; published directories do not require our custom dropout-head checkpoint format.

## Evaluation protocol

- Use Gaussian noise and brightness for development; reserve blur/test images for final evaluation.
- Accepted protocol: retain the current development manifest, which uses CIFAR-10 training images. Published classifiers may already have seen these images. Disclose this potential overlap in the report and presentation as a limitation; do not describe these development results as classifier-held-out performance or assume external authors used our 45k/5k split. Warning-model fitting/calibration and final evaluation remain separated, with reserved test images and blur excluded from development. External checkpoint exposure to the official test set is not independently established.
- Select clean-correct images independently for each classifier, and report cohort differences across classifiers. Within each classifier, warning methods share examples and image-grouped cross-validation folds across both corruptions.
- Evaluate next-severity failure ranking with AUROC/AUPR; select thresholds using development data and compare warning lead time at matched false-alarm rates on held-out data.
- Fit L/G/D logistic combinations and component ablations separately for IG-derived and Grad-CAM-derived maps. These are learned combinations, not a claim that a particular fixed warning-score formula has been validated.
- Document clean-reference access, failures, false alarms, missed warnings, and computational cost.

ResNet Grad-CAM uses the final bottleneck convolution; ViT uses the last pre-attention LayerNorm with class-token removal; Swin uses the last block's pre-attention LayerNorm with spatial token reshaping. Heatmaps return to original image dimensions.

See [PROJECT_PLAN.md](../PROJECT_PLAN.md) for progress and historical pilot results.

```bash
python -m unittest discover -s tests -v
```

## Frozen final blur evaluation

Development-only selection is now fixed: choose the attribution variant with the greatest mean credited lead over all failures among candidates whose pooled development-CV trajectory false-alarm rate is at most 5%. Ties prefer early-warning rate, then model name; if no candidate qualifies, stop rather than relax the budget. This selection is exploratory development model selection, not an unbiased performance estimate.

Selected variants: **ResNet-50: Grad-CAM L+D fixed; ViT-B/16: Grad-CAM L+D fixed; Swin-T: Grad-CAM L raw**. A deterministic image-group split uses 75% of development image identities for final score fitting/scaling and 25% for threshold calibration; both corruption trajectories remain together. Thresholds for MSP, ODIN, and the selected score are calibrated only on nonfailing development trajectories. Models, scales, thresholds, split identities, source-data hash, collection settings, and checkpoint provenance are saved in `results/frozen_warning_methods/<classifier>.json`. Existing frozen artifacts cannot be overwritten by the freeze command.

```bash
python -m scripts.freeze_warning_methods
python -m scripts.run_final_evaluation
```

The freeze command has already been run. Do not rerun it to change settings after viewing final results. Final collection uses all clean-correct images from the existing reserved 500-image test manifest, rather than forcing a per-class quota; each classifier's selected count may differ. `selection_audit.csv` records every reserved candidate and clean prediction, including clean-incorrect exclusions. Only blur severities 0–5 are collected. Results go to `results/final_blur_evaluation/<classifier>/evaluation/`; progress is in `results/final_blur_evaluation_run.log`. No final performance is claimed until collection and evaluation finish.

The standalone `scripts/evaluate_frozen_warnings.py` applies JSON score parameters and thresholds without fitting, scaling updates, threshold tuning, or variant selection. It rejects development-source data, non-blur data, collection-setting mismatches, and image identities outside the reserved test manifest. Final outputs include early warnings, misses, credited lead, achieved false alarms, paired baseline comparisons, exclusions, and a hash/copy of the frozen artifact. Test false-alarm rates are measured, not forced to match calibration targets.

## Exploratory warning improvements

After viewing the original final blur results, a separate **development-only exploratory** experiment tests:

- Unweighted Wasserstein distance and attribution mass loss `1 - retention` as separate features, including a low-retention proxy (`retention <= 1e-6`). Historical zero distances for vanished maps are not reinterpreted as valid transport distances; the proxy does not reconstruct the original exact cutoff.
- Consecutive-severity changes in L, G, and D. These require sequential observations and use only the current and previous severity; they are not time derivatives.
- Learned separate-distance/retention combinations, current-plus-trend combinations, and confidence-plus-attribution combinations.
- Confidence-trend controls alongside unchanged MSP/ODIN and original attribution candidates.

```bash
python -m scripts.test_warning_improvements
```

The run reads only `results/classifier_benchmarks_500/*/images.csv` (training-source Gaussian noise and brightness). It uses the same nested image-grouped fitting/calibration/evaluation protocol at nominal trajectory false-alarm targets of 1%, 5%, and 10%. Report achieved evaluation false-alarm rates and pairwise timing results; a higher warning count at a higher false-alarm rate is not automatically an improvement. All fitting transformations use fitting images only. Whole invalid trajectories are excluded consistently across methods.

Outputs: `results/exploratory_warning_improvements/`; log: `results/exploratory_warning_improvements_run.log`. The script refuses to overwrite an existing experiment directory. Original frozen methods and final blur results remain unchanged. Improvements discovered here are exploratory and need fresh confirmation; the already-viewed blur results are not a new untouched validation set. Competing-class attribution and logit-margin features are not included because the stored CSVs do not contain the required competing logits/maps.

### Isolating attribution's contribution

`python -m scripts.analyze_attribution_contribution` analyzes the existing exploratory out-of-fold trajectories without fitting or recollecting data. It compares attribution-only trends, confidence-plus-confidence-trend, and confidence-plus-attribution/trends on identical trajectories, separately for IG and Grad-CAM. MSP and ODIN results remain in the underlying experiment.

Outputs: `results/attribution_contribution_analysis/FINDINGS.md` and `paired_effects.csv`. Paired percentile 95% confidence intervals use 2,000 image-cluster bootstrap draws, resampling both corruptions together. Results cover all three classifiers, individual corruption types, and 1%, 5%, and 10% calibration targets. Intervals are conditional on the existing fitted models/thresholds, do not include refitting variability, and do not adjust for multiple candidate comparisons. False-alarm-rate differences are reported alongside detection and lead-time differences.

Initial interpretation: attribution adds development warning information beyond confidence trends for ResNet-50 and Swin-T, with false-alarm trade-offs; this is not consistent for ViT-B. These exploratory findings need fresh confirmation and do not replace the original frozen final results. No expansion to more images has been started by this analysis.

## Revised warning score: fresh-image confirmation

The revised trend-based methods are frozen in `results/fresh_trend_validation/frozen/`, independently of the original artifacts. Before fresh-image predictions, `locked_protocol.json` records hashes of these artifacts and the new manifest. The primary comparison for each classifier is **confidence + Grad-CAM attribution/trends versus confidence + confidence trend**. IG combined and attribution-only variants are prespecified secondary comparisons; MSP and ODIN remain reference baselines. All seven scores are calibrated to the same 5% development trajectory false-alarm target. Final logistic weights, feature scalers, thresholds, checkpoint provenance, and image-group fitting/calibration splits are exported to JSON.

The revised combined features are current unconfidence and its previous-severity change, L, G, unweighted W, `1-retention`, a low-retention proxy, and previous-severity changes in L/G/D. No new image gradients are needed for feature construction beyond the usual IG/Grad-CAM collection. Next-step failure labels fit the learned model on development data; no test labels fit it. Both corruption trajectories of every image stay in one development role. A 75% fitting / 25% calibration image split is fixed with seed 42.

A deterministic fresh manifest reserves **500 candidates, 50 per class**, from the remaining official CIFAR-10 test images. It excludes 525 possible prior test-image IDs found in earlier manifests/results, including candidates excluded by previous clean-accuracy checks and conservatively treating source-less old pilot IDs as possible exposure. Evidence hashes and the full exclusion list are saved in `reservation_audit.json`. Selection uses only labels and seed 20261101, without classifier predictions or warning scores. As before, exposure during external classifier training is not independently established.

Setup is complete and should not be rerun to change choices after seeing outcomes:

```bash
python -m scripts.fresh_trend_validation setup
python -m scripts.fresh_trend_validation run
```

The run is launched. Progress: `results/fresh_trend_validation_run.log`. Collection and `evaluation/` outputs are under `results/fresh_trend_validation/<classifier>/`. All clean-correct candidates are evaluated; clean-incorrect candidates and invalid trajectories are audited. Existing experiments cannot be overwritten. Evaluation validates cohort membership/completeness and locked settings, and applies exported parameters without fitting or recalibration. It reports paired image-bootstrap confidence intervals against confidence trends, MSP, and ODIN.

This is **fresh-image confirmation under blur, a previously examined corruption**, not confirmation on an unseen corruption type. The protocol does not authorize test-based threshold changes or post-result variant selection. Achieved test false-alarm rates must be reported; a higher warning rate at a higher false-alarm rate is not automatically superiority.

## Fresh-image secondary AUROC and AUPR

`python -m scripts.evaluate_fresh_ranking` computes descriptive next-step ranking metrics from the same frozen revised scores, without fitting, changing thresholds, or selecting models. Outputs: `results/fresh_trend_ranking/summary.csv`, per-classifier prediction/ROC/PR CSVs, and `roc_pr_curves.png`.

The population is currently correct, pre-first-failure observations at severities 1–4. A positive label means first failure at the next severity. Recoveries after first failure are excluded. Failures already at severity 1 have no preceding monitored next-step example, so ranking sample counts differ from primary trajectory counts. AUPR is sklearn **average precision (AP)**, not trapezoidal PR integration. ROC false-positive rates here are per next-step example, not the primary trajectory-level false-alarm rate. AP should be interpreted alongside failure prevalence, which differs across classifiers.

This is a post-hoc secondary analysis of completed fresh-image results, not new confirmation or a basis for retuning. ResNet-50 combined confidence/Grad-CAM trends score: AUROC 0.749, AP 0.684 (MSP 0.717/0.622; ODIN 0.608/0.483). ViT and Swin ranking favors MSP in AUROC and confidence trends in AP. Warning timing remains primary; ranking and a fixed-threshold early-warning operating point need not yield the same ordering. All 24 automated tests pass.

## Development severity-spacing sensitivity pilot

`scripts/study_severity_spacing.py` collects 50 balanced, previously clean-correct development images per classifier (five per class) under noise/brightness at 20 nonzero physical strengths plus clean. It subdivides each original interval into four; subsampling by factors two/four yields nested 10/5-strength grids that include every original endpoint. All grids use the same image cohort, original maximum corruption strength (noise std 0.20, brightness offset 0.40), and Gaussian noise direction. Existing integer-severity code and historical results remain unchanged; `apply_corruption_strength` is a separate continuous helper.

```bash
python -m scripts.study_severity_spacing --n-per-class 5
```

Collection is launched; progress: `results/severity_spacing_study_run.log`. Outputs: `results/severity_spacing_study/`. No test images, frozen methods, or prior evaluations are modified.

The densest 20-strength grid defines the common first-failure reference for all resolutions; this is still an approximation to the actual corruption failure point. Trend features divide by increments of normalized physical strength `c = strength / original maximum`. Learned warnings predict first failure within a fixed horizon `h = 0.2` in c, rather than at the next sampled point. Thus the task horizon does not shrink as sampling becomes denser. This differs from the previous next-step task and is reported as a separate sensitivity experiment.

Five outer image-grouped folds and disjoint inner fit/calibration groups are shared across grids. The 5% trajectory false-alarm target is recalibrated separately for each resolution and alarm rule, since denser monitoring creates more alarm opportunities. MSP, ODIN, confidence derivatives, and combined IG/Grad-CAM feature/derivative scores are compared. Single exceedance and two-consecutive rules are both exploratory conditions here, not replacements for the previously frozen system. Report skipped learned folds if fitting labels lack both classes, and calibration negative counts because this pilot has limited threshold precision.

Primary lead units are normalized corruption-strength distance, with zero credited lead for misses. Raw physical-strength lead is reported per corruption only; noise-standard-deviation and brightness-offset units are not pooled. No lead is reported as a mere number of sampled levels. A finer grid can reveal failures between old observations, so a shared dense failure reference and common denominators are used. This pilot tests sampling sensitivity, not fresh-image or unseen-corruption generalization.

### ResNet-only severity analysis

At the user's request, the ongoing collection was interrupted during ViT-B to analyze the completed ResNet-50 pilot first; partial transformer data is retained. Analyze completed classifiers independently with:

```bash
python -m scripts.study_severity_spacing --analyze-only --classifiers resnet50 --analysis-output results/severity_spacing_resnet_analysis
```

This analysis has completed; use a new output directory to rerun without overwriting it. Findings: `results/severity_spacing_resnet_analysis/FINDINGS.md`. At 20 strengths, single-alarm early-warning rates were 52.8% for confidence+Grad-CAM, 54.7% for confidence+IG, 54.7% for MSP, and 62.3% for confidence derivatives. Actual false-alarm rates were 10.6%, 8.5%, 6.4%, and 8.5%, respectively. Denser sampling helps detect warnings, especially for confidence baselines; it does not establish an attribution advantage. This is a 50-image development pilot with shared dense-grid failure denominators and fixed strength horizons, not a new final test.

## Exploratory learned IG + Grad-CAM fusion

`python -m scripts.test_ig_gradcam_fusion` tests whether concatenating both attribution generators' custom features helps. It uses existing ResNet-50 development data only, with identical nested image splits and 5% calibration false-alarm targets. It compares static fusion, fusion with L/G/D trends, and confidence-plus-fusion/trends against individual-map learned combinations and confidence controls. There is no pixelwise heatmap averaging and no change to frozen/final methods.

Results: `results/resnet_ig_gradcam_fusion/FINDINGS.md`. Confidence+fusion warned before 187/515 failures (36.3%), compared with 173/515 for confidence+Grad-CAM and 162/515 for confidence+IG. Actual false-alarm rates were 5.8%, 6.0%, and 7.8%, respectively. These are exploratory development estimates, not confirmed fresh-data gains; paired bootstrap intervals are saved in `fusion_paired_effects.csv`. All 30 tests pass.

## ResNet IG + Grad-CAM fusion fresh confirmation

The learned fusion is now frozen independently in `results/fresh_fusion_confirmation/frozen/resnet50.json`. All weights/scalers fit development images only; separate development calibration sets each method's 5% trajectory false-alarm threshold. The primary comparison is confidence+IG+Grad-CAM feature/trend fusion versus the individual confidence+Grad-CAM combination. Confidence+IG, confidence trends, MSP, and ODIN are additional controls; attribution-only fusion is secondary. No test-based model or threshold adjustment is permitted.

```bash
python -m scripts.confirm_fusion_fresh setup
python -m scripts.confirm_fusion_fresh run
```

Setup is complete and the run is launched. A newly reserved 500-image test candidate cohort (50/class, seed 20261103) excludes all earlier recorded test candidates/observations, including clean-incorrect exclusions. `reservation_audit.json` records evidence and excluded IDs; `locked_protocol.json` hashes the manifest and frozen artifact before new predictions. The classifier evaluates all clean-correct fresh candidates under blur severities 0–5. This is a ResNet-only confirmation on fresh images under previously examined blur, not an unseen-corruption experiment. External classifier training exposure remains unverified/disclosed.

Progress: `results/fresh_fusion_confirmation_run.log`. Collection and timing results: `results/fresh_fusion_confirmation/resnet50/`; AUROC/AP and ROC/PR CSVs follow under `evaluation/secondary_ranking/`. Paired image-bootstrap effects compare fusion against individual generators and confidence controls, including achieved false-alarm-rate differences. Existing frozen experiments and results are preserved. All 30 tests pass before launch.
