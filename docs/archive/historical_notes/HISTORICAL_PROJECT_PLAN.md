> Current scope update: Group DRO and V-REx are discontinued from active development. Entries below describing those runs are historical records. Next focus is development-only L/G/D feature diagnosis, followed by evidence-led feature design. No further DRO/V-REx runs are planned.

# Final Project Progress Tracker

## Current research scope — corrected attribution-only comparison

The proposed warning scores use only custom attribution features and their trends from IG, Grad-CAM, or learned fusion of both. Classifier confidence, confidence trends, MSP, and ODIN are competing baselines, never inputs to an attribution score.

- [x] Remove confidence–attribution combinations from active development, fusion, dense-grid, and fresh-protocol builders.
- [x] Reject confidence/label/severity inputs in attribution model fitting and role-tagged prediction.
- [x] Make future fresh protocols attribution-only, with versioned output folders and guards against launching old hybrid protocols as corrected experiments.
- [x] Verify attribution scores are invariant to confidence-column changes; all 34 tests pass.
- Historical hybrid artifacts/results below remain unchanged and must not be used to claim standalone attribution superiority.
- The previously reported favorable confidence+attribution results are hybrid findings only. Standalone attribution has not established consistent superiority over confidence.

## Historical work log (some entries describe superseded hybrid experiments)


## Research question and finalized evaluation plan

Can attribution changes warn of classification failure earlier than model confidence, at comparable false-alarm rates?

Start with a small development run, then scale up once the pipeline is verified. Develop using Gaussian noise and brightness. Reserve test images and blur corruption for final evaluation; do not use them to fit models or select thresholds.

### Core baselines

The revised project evaluates **ResNet-50, ViT-B, and Swin-T** against these warning baselines:

- **MSP:** model confidence.
- **ODIN:** temperature-scaled, input-perturbed confidence.

Energy and corruption severity belong to the historical pilot, not the revised core baseline set. The revised classifier loader, benchmark scores, runner, and comparison are implemented; published CIFAR-10 checkpoints are downloaded and supported; MC Dropout has been removed from the benchmark set. See [README.md](../README.md) for the current project scope.

- [x] Add ResNet-50, ViT-B/16, and Swin-T loader, preprocessing, and attribution reshaping
- [x] Download and integrate published CIFAR-10 checkpoints without further fine-tuning
- [x] Implement ODIN baseline (development parameter selection pending)
- [ ] Run the revised comparison across all three classifiers

### Proposed attribution methods

IG and Grad-CAM generate the attribution maps; the proposed warning signals are L, G, and D calculated from those maps. Historical results below use logistic regression to combine these metrics and do not establish the performance of other warning-score formulas.

- IG warning model using attribution loss (L), gain (G), and retention-weighted Wasserstein distance (D).
- Grad-CAM warning model using the same components.
- Confidence plus IG metrics, and confidence plus Grad-CAM metrics.
- Evaluate individual components and other combinations during Week 3 ablations.

### Previously considered extensions (outside the current core scope)

- Mahalanobis distance: fit class means and a shared, regularized covariance on a separate clean reference subset. A penultimate-layer, nearest-class implementation is a simplified variant of [Lee et al.](https://arxiv.org/abs/1807.03888).
- Clean-reference feature distance: compare each corrupted embedding with its own clean embedding, to assess the benefit of attribution changes beyond embedding changes.

### Shared evaluation rules

- Use the same eligible images and severity transitions for all methods; report any exclusions.
- Secondary next-step task: among clean-correct images still classified correctly at a nonzero current severity, predict incorrect classification at the next sampled severity. Keep corruption trajectories separate.
- Keep every image's severities and corruption variants together when splitting development data for fitting and validation.
- Report next-step AUROC and AUPR, sample counts, and failure prevalence.
- Select warning thresholds on development data at comparable false-alarm rates; freeze them before final testing.
- Compare first warning severity and lead time relative to first failure. Report missed warnings and false alarms; severity steps are not elapsed time.
- Document that attribution comparisons and clean-reference feature distance require access to a clean reference image.

## Published-checkpoint integration

- Pinned checkpoint sources and revisions: `configs/hf_classifiers.json`; loader: `src/classifiers.py`.
- ResNet-50: `edadaltocg/resnet50_cifar10`; ViT-B/16: `jialicheng/cifar10_vit-base`; Swin-T: `jcollado/swin-tiny-patch4-window7-224-finetuned-cifar10`.
- All three passed a 20-image development prediction check and finite IG/Grad-CAM/ODIN checks; see `results/published_checkpoint_smoke.json`. This is not a held-out accuracy estimate.
- End-to-end one-image smoke runs passed for all three classifiers: 24 rows each, Gaussian noise and brightness at severities 0–5, finite attribution/MSP/ODIN scores, zero clean attribution differences, and explicit handling of unavailable dropout in the historical smoke output. Outputs: `results/published_smoke/`.
- A 50-image-per-classifier development collection/comparison was launched with `python -m scripts.run_classifier_benchmarks --n-per-class 5`. Progress: `results/published_benchmark_run.log`; outputs: `results/classifier_benchmarks/`. Completion and full warning results are not yet verified.
- Accepted decision: keep the current development images despite potential exposure during external classifier training. Disclose this overlap as a limitation in the report and presentation, and do not claim development images are held out from classifier training. Continue separating warning-model development/calibration from reserved final evaluation; external checkpoint exposure to the official test set is not independently established.

## Week 2: Build robust baselines

- Status: **in progress**. Pilot code exists; baseline validation and consistent evaluation remain unfinished.
- Current runner: `scripts/run_asws_experiment.py` collects MSP, ODIN, and attribution metrics for a specified classifier/checkpoint. Energy is removed from active collection and comparison.
- Run all three classifiers and their comparisons with `python -m scripts.run_classifier_benchmarks --n-per-class 5` using the downloaded published checkpoints. See README for details.
- The completed checkboxes and results below refer to the historical pilot, not completion of the revised benchmark.
- Outputs: `results/development_asws/` and `results/development_comparison/`. Small-run results are pipeline checks, not final evidence of superiority or earlier warnings.
- [x] Validate raw model-confidence (MSP) baseline on the small development run
- [x] Implement and validate energy-score baseline on the small development run
- [x] Corruption-severity baseline on the small development run
- [x] Run Gaussian-noise and brightness development experiments (50 images, 1,200 rows)
- [x] Integrated Gradients (IG) warning model: initial grouped cross-validation
- [x] Grad-CAM warning model: initial grouped cross-validation
- [x] Combined models: initial grouped cross-validation

### Initial development results

Historical pilot only: these results use the previous classifier and baseline set, not the revised ResNet-50 / ViT-B / Swin-T benchmark.

Expansion run launched: 500 clean-correct development images (50 per class), with the same two corruptions and severities 0–5. Expected output: 12,000 rows in `results/development_asws_500/images.csv`. The comparison is queued to run after successful collection and save to `results/development_comparison_500/`. Progress log: `results/development_500_run.log`. Completion and expanded results have not yet been verified.

The completed run has 287 eligible next-severity examples, including 49 failures (41 under Gaussian noise and 8 under brightness). No examples were excluded for invalid scores. Learned models use five shared folds grouped by image across corruption types.

| Method | Pooled AUROC | Pooled AUPR |
|---|---:|---:|
| Raw MSP warning | 0.893 | 0.668 |
| Raw energy | 0.852 | 0.647 |
| Raw severity | 0.420 | 0.153 |
| IG (L, G, D) | 0.809 | 0.444 |
| Grad-CAM (L, G, D) | 0.854 | 0.573 |
| Confidence + IG | 0.847 | 0.630 |
| Confidence + Grad-CAM | 0.855 | 0.628 |

These exploratory results do not establish attribution superiority or earlier warnings. Inspect per-corruption results in `results/development_comparison/summary.csv`; false-alarm-matched thresholds, lead time, uncertainty estimates, and final held-out evaluation remain outstanding. Week 2 has an initial working comparison; robustness checks and the decision to scale up remain pending.

## Week 3: Improve the warning score

- [ ] Evaluate attribution loss (L), attribution gain (G), and spatial distance (D) contributions
- [ ] Run ablation experiments
- [x] Implement nested image-grouped fit/calibration/evaluation for development warning thresholds (5% trajectory false-alarm target)
- [ ] Finalize threshold policy after robustness checks; do not tune on reserved test images

## Week 4: Evaluate the finalized method

- [x] Freeze classifier-specific attribution score, fitting-only scales, MSP/ODIN and attribution thresholds from development data
- [ ] Evaluate on the reserved test images with held-out blur corruption
- [ ] Compare the finalized method against the baselines

## Week 5: Analyze warning behavior

- [ ] Analyze failures
- [ ] Analyze false alarms
- [x] Implement primary first-failure timing evaluation, missed warnings, all-failure credited lead, trajectory false alarms, and paired baseline comparisons
- [ ] Interpret timing results across classifiers and achieved false-alarm rates

## Week 6: Complete final deliverables

- [ ] Complete the research report
- [ ] Finalize results, tables, and figures
- [ ] Discuss limitations
- [ ] Disclose potential overlap between development images and external classifier training data
- [ ] Complete the presentation
- [ ] Prepare reproducibility materials

## Primary evaluation enforced

`scripts/fit_aws_logreg.py` now produces first-failure timing as its primary summary and writes AUROC/AUPR under `secondary_next_step_*`. The timing evaluator includes raw L/G/D and fixed pair/full ablations for each map generator, plus learned full combinations. Fitting, threshold calibration, and outer evaluation use disjoint image groups. Misses receive zero credited lead; alarms at/after failure do not count as early. Thresholds target 5% calibration trajectory false alarms; achieved evaluation false alarms must be reported and may differ.

ResNet-50 timing was evaluated on the completed 50-image dataset, with 100 corruption trajectories (54 fail, 46 never fail). No trajectory exclusions. Results: `results/classifier_benchmarks/resnet50/comparison/early_warning/`. ViT/Swin comparisons will run the revised evaluator when collection finishes. This is still development analysis; the final reserved evaluation remains untouched.

## Final protocol frozen; reserved blur run launched

- Development selection rule: maximize all-failure credited lead among attribution candidates with pooled development-CV false alarms <= 5%; tie-break by early-warning rate then name.
- Selected scores: ResNet-50 and ViT-B/16 use Grad-CAM L+D fixed; Swin-T uses Grad-CAM L raw.
- Artifacts saved before opening reserved blur data: `results/frozen_warning_methods/`; final fitting/calibration groups, thresholds, and provenance are recorded.
- Final runner launched: `scripts/run_final_evaluation.py`, using all clean-correct images in the reserved test manifest and blur only. Candidate exclusions are audited.
- Progress: `results/final_blur_evaluation_run.log`. Final performance and completion remain unverified until outputs are available.
- Twelve automated tests passed before launch, including frozen-score serialization and rejection of development data by the final evaluator.

## Exploratory feature improvements

- [x] Implement development-only separate W/retention features and causal consecutive-severity L/G/D trends.
- [x] Add confidence-trend controls and preserve original methods/folds for comparison.
- [x] Validate causal features and unchanged original predictions/splits; all 14 automated tests pass.
- [x] Complete exploratory runs at 1%, 5%, and 10% calibration false-alarm targets for all three classifiers.
- [x] Reproduce original 5%-target results on shared cohorts; record initial findings in `results/exploratory_warning_improvements/FINDINGS.md`.
- [ ] Confirm exploratory trend improvements with a preregistered revised protocol and fresh evaluation data; account for confidence-trend controls and achieved false-alarm differences.
- Outputs: `results/exploratory_warning_improvements/`; progress: `results/exploratory_warning_improvements_run.log`.
- This experiment was designed after viewing original final results. No frozen artifact or original blur result is modified. Fresh confirmation is needed for any revised method.

## Attribution contribution analysis completed

- [x] Compare attribution-only, confidence-trend, and combined learned warning scores on existing shared development folds.
- [x] Add paired image-cluster bootstrap intervals (2,000 draws) for early-warning rate, all-failure credited lead, and false-alarm-rate differences.
- [x] Cover all classifiers, both corruptions separately/pooled, and 1%, 5%, and 10% calibration targets.
- Findings: `results/attribution_contribution_analysis/FINDINGS.md`; full effects: `paired_effects.csv`.
- Combined attribution/confidence trends improve development detection and lead on ResNet-50 and Swin-T, with false-alarm trade-offs; no consistent benefit on ViT-B. Intervals condition on existing fitted runs and are not multiple-comparison-adjusted.
- All 17 automated tests pass. No new image collection, frozen-artifact changes, or original test-result changes.
- [ ] Choose a revised development candidate and fresh confirmation protocol before scaling or making new generalization claims.

## Revised score frozen and fresh-image run launched

- [x] Freeze MSP, ODIN, confidence trends, IG/Grad-CAM attribution trends, and combined scores using development data only.
- [x] Declare the primary comparison (confidence + Grad-CAM attribution/trends versus confidence trends) and IG secondary comparisons before fresh outcomes.
- [x] Reserve 500 new official-test candidates, 50/class, excluding 525 possible prior test IDs; preserve old manifests/artifacts/results.
- [x] Lock artifact/manifest hashes, feature definitions, thresholds, fitting/calibration split, and checkpoint provenance.
- [x] Pass 21 automated tests, including an end-to-end no-test-fitting/no-test-calibration check.
- [x] Launch `scripts.fresh_trend_validation run` for all three classifiers.
- [ ] Verify completed fresh-image results and interpret paired warning/lead/false-alarm differences without retuning.
- Progress: `results/fresh_trend_validation_run.log`; artifacts, manifest, reservation evidence and outputs: `results/fresh_trend_validation/`.
- Blur has already been examined. This tests fresh-image transfer, not unseen-corruption transfer; external classifier training exposure remains disclosed.

## Revised fresh-image AUROC/AUPR completed

- [x] Calculate secondary next-step AUROC and average precision (AUPR/AP) for all seven frozen methods and all three classifiers.
- [x] Export per-example scores, ROC/PR curve coordinates, and a combined report-ready figure.
- Outputs: `results/fresh_trend_ranking/summary.csv` and `roc_pr_curves.png`; runner: `scripts/evaluate_fresh_ranking.py`.
- No models, thresholds, frozen protocol files, or primary timing results changed. This is post-hoc descriptive reporting.
- Next-step ranking differs from first-failure timing: severity-1 failures have no eligible prior monitored step, and ROC FPR is per example rather than per trajectory.
- All 24 tests pass. Combined Grad-CAM trends outperform the ranking baselines on ResNet-50; results are mixed across architectures.

## Severity-spacing sensitivity pilot launched

- [x] Implement a separate continuous noise/brightness helper preserving original corruption ranges and integer-severity behavior.
- [x] Build nested 5/10/20 nonzero-strength grids, retaining every original strength.
- [x] Define strength-normalized trends, a fixed 0.2 normalized-strength prediction horizon, and strength-based warning lead.
- [x] Validate 28 automated tests plus an end-to-end synthetic analysis (shared failure denominators and image roles across grids).
- [x] Launch 50 development images per classifier, IG and Grad-CAM, noise and brightness, same noise direction across strengths.
- [ ] Complete collection and compare warning/detection/false-alarm trade-offs across grids; interpret small pilot calibration uncertainty.
- Progress: `results/severity_spacing_study_run.log`; outputs: `results/severity_spacing_study/`.
- Original static/trend frozen artifacts, completed test results, and integer-severity experiments are unchanged. Persistence is an exploratory grid-sensitivity condition, not a change to the existing alarm rule.

## ResNet-only severity-spacing analysis completed

- User narrowed scope to ResNet-50. Interrupted the running collector during ViT-B; partial transformer data is preserved, Swin-T has not run.
- [x] Analyze completed 50-image ResNet collection with nested 5/10/20 strength grids.
- Findings: `results/severity_spacing_resnet_analysis/FINDINGS.md`.
- Finer spacing improves observed detection for attribution and confidence scores. At 20 strengths the confidence-only derivative control detects more failures than either combined attribution variant; no clear attribution win in this pilot.
- No original frozen models, thresholds, completed test results, or existing alarm rules changed.

## Exploratory IG + Grad-CAM fusion completed

- [x] Test learned concatenation of both attribution generators' static features, trends, and confidence on existing ResNet-50 development data.
- [x] Use shared image splits/calibration and compare with individual-map and confidence-trend controls.
- Results: `results/resnet_ig_gradcam_fusion/FINDINGS.md`; paired image-bootstrap effects saved alongside results.
- Confidence+fusion improves point estimates over individual-map combinations: 187 vs 173 vs 162 early warnings out of 515 failures, with 5.8% vs 6.0% vs 7.8% actual false alarms.
- No new image collection or frozen/test changes. The fusion is exploratory and not fresh-data validated. All 30 tests pass.

## Fresh fusion confirmation launched

- [x] Freeze ResNet-50 fusion, individual-generator combinations, confidence trends, MSP and ODIN from development data only.
- [x] Reserve another disjoint 500-candidate test cohort and lock artifact/manifest hashes before predictions.
- [x] Declare primary fusion-vs-individual-Grad-CAM comparison; keep other generators/confidence/MSP/ODIN as controls.
- [x] Validate protocol-specific paired comparisons and all 30 tests; launch ResNet-only collection on fresh blur images.
- [ ] Complete fresh fusion timing evaluation and secondary AUROC/AP; interpret warning gains jointly with false alarms.
- Progress: `results/fresh_fusion_confirmation_run.log`; outputs: `results/fresh_fusion_confirmation/`.
- Original/fresh previous results remain unchanged. Blur is known, and this is fresh-image confirmation with no test refitting or threshold retuning.

## Attribution-only robust-warning training pilot completed

- [x] Implement pooled ERM, group-balanced ERM, and convex minimax corruption-group DRO with identical regularization and attribution-only predictors.
- [x] Run controlled ResNet-50 development comparisons for IG, Grad-CAM, and fusion, using shared image-group splits/calibration and confidence baselines separately.
- [x] Save group log loss, timing/false alarms, paired bootstrap effects, and fold model parameters.
- Results: `results/resnet_warning_erm_dro/FINDINGS.md`.
- Group DRO improves worst observed-corruption OOF logistic loss in all three recipes, but warning-rate gains are not consistent (small Grad-CAM improvement, IG/fusion regressions).
- All 38 tests pass. Classifier weights, original frozen scores, and test results remain unchanged.
- [ ] Design a multi-corruption development dataset and leave-corruption-out validation before claiming OOD generalization improvements. The two-group pilot does not evaluate an unseen corruption.

### V-REx development pilot

- Completed attribution-only ResNet V-REx vs ERM/balanced ERM on the existing 500-image development cohort; beta=1 fixed, no test tuning.
- Grad-CAM false alarms improved (5.98% to 4.74%); early warnings 132 to 137, with uncertain detection improvement. IG and fusion did not improve overall.
- Results: `results/resnet_warning_vrex/FINDINGS.md`. Pending: additional development corruption families and held-out-family validation.

### Frozen V-REx fresh-image confirmation

- Locked Grad-CAM ERM, balanced ERM and V-REx (beta=1, L2=.01) using development fitting/calibration images only. MSP, ODIN and separate confidence trends included.
- Reserved 500 label-stratified test candidates, excluding all recorded prior test candidates; clean-correct selection precedes attribution collection.
- Started `scripts.confirm_vrex_fresh run`; root `results/fresh_resnet_vrex_confirmation/`, progress `run.log`. Automatic paired timing evaluation and secondary AUROC/AP after collection.
- Blur was previously examined; confirmation tests fresh images under known blur, not a new corruption family. No test refitting or threshold retuning.

### Attribution feature diagnosis

- Run descriptive diagnosis on existing 500-image development CSVs for ResNet, ViT-B and Swin-T.
- Compare next-step failures, later observed failures and nonfailing trajectories at matched severity and corruption. Exclude post-failure rows and clean observations.
- Save raw-direction AUC within severity, severity association among nonfailing trajectories, group counts and median/IQR plots. No fitting, test access or frozen-method changes.
- DRO/V-REx code and outputs retained only for reproducibility; see `docs/DISCONTINUED_ROBUST_WARNING_EXPERIMENTS.md`.

### Attribution interaction development pilot

ResNet development-only matched comparison completed. Grad-CAM L/G/D+trends interactions detected 161/515 failures early versus 137 additive, at 5.57% versus 5.98% false alarms. Adding retention gave 158 early warnings and 4.95% false alarms. IG interactions also improved detection, with higher false alarms than Grad-CAM. No confidence inputs, test access or frozen-score changes. These reused-development results require independent validation. See [findings](../results/resnet_attribution_interactions_v2/FINDINGS.md).

### Cross-corruption interaction check

Completed the ResNet source-only fit/calibration check in both directions with image-disjoint evaluation. Interactions did not establish transferable warning gains: Grad-CAM noise→brightness warned 4/56 early (4.50% false alarms); brightness→noise warned 90/459 early (80.49% false alarms). Confidence baselines also lost calibration in the latter direction. No target recalibration or test access. Further classifiers were not launched because the transfer gate did not pass. See [findings](../results/resnet_corruption_transfer/FINDINGS.md).
