> **Protocol update:** The final run now uses official CIFAR-10-C files. See [official protocol](docs/CIFAR10C_PROTOCOL.md). Previous custom-corruption results remain historical and must not be presented as CIFAR-10-C benchmark results. New run: `results/final_cifar10c/`; the stopped custom run is superseded.

# Presentation and report completion

Method expansion is closed. Prepare next week's presentation using the [presentation guide](docs/PRESENTATION_GUIDE.md).

- [x] Core attribution-only pipeline, original datasets/splits and checkpoint provenance preserved.
- [x] New IG setting: powers of two 1–128, default 64; historical 50-step evidence preserved.
- [x] Primary corruption protocol: severity levels 0–5.
- [x] Stability, convergence and completeness-check scripts/evidence grouped under validation.
- [x] Original frozen evaluations consolidated in [Week 4 report](docs/WEEK4_EVALUATION.md).
- [x] Final attribution/error figures and association tables prepared for all three classifiers.
- [x] Exploratory methods, results and historical notes archived; no data deleted.
- [ ] Select presentation figures and write captions with model/cohort/score specifications.
- [ ] Complete report discussion, failure examples and limitations.
- [ ] Prepare slides and rehearse the distinction between error detection and early warning.

Attribution carries predictive information, but reliable superiority over confidence under corruption shift remains unproven. Preserve this distinction rather than combining incompatible experiments. See [archive index](docs/ARCHIVE_INDEX.md) for historical work.

## Authorized final rerun

Running `scripts/run_final_project.py` with a locked 500-development/200-fresh-test-candidate protocol, IG64, original normalization, L/G/D+trends primary IG and Grad-CAM logistic models, static and interaction ablations and separate confidence baselines. Model parameters and thresholds are preserved before test evaluation. All three architectures are scheduled sequentially within an approximately 10-hour resumable budget. See `results/final_project_run/run.log` and `status.json`. Fresh image IDs do not make previously examined blur untouched.

## Supplementary confidence baselines

Entropy (Shannon, 1948) and post-hoc Energy (Liu et al., NeurIPS 2020) are added as separately locked supplementary comparisons after the primary official run started. Both use fixed temperature 1, the original calibration image groups and identical common-valid test cohorts; no attribution recomputation or test tuning. See [formulas, citations and execution](docs/SUPPLEMENTARY_BASELINES.md). The primary protocol and existing outputs remain unchanged.

## Active attribution method restored

Use the original **L/G/D attribution-only pipeline**: IG and Grad-CAM L/G/D plus their existing severity-to-severity trends as primary warning models, with original static/interaction ablations and frozen development calibration. Pearson, cosine, projections, sparse encodings and their combinations are discontinued and [archived](results/archive/map_similarity/README.md). Keep the established confidence baselines, including separately documented entropy/Energy supplements. The official CIFAR-10-C run and its frozen artifacts remain authoritative.

## Expanded sample-size follow-up

The original official 500-development/200-test-candidate study completed in **1h39m32s** on one A100; preserve/report it. A separately locked [expanded run](results/expanded_cifar10c_5000dev_2000test/README.md) uses **5000 development candidates and2000 additional test candidates**, retaining all clean-correct/valid images, with original IG64 L/G/D methods. New models/thresholds are fitted on expanded development data only. Slurm22982030 requests48hours; estimated runtime15–20h. This is a follow-up after prior results, not untouched-corruption confirmation. No similarity/encoding extensions are active.

### Entropy/Energy on the expanded cohort

The supplement now targets the **same expanded5000/2000 study**, not the original500-image study. Entry point: `scripts/evaluate_expanded_supplementary_baselines.py`. Slurm **22982067** is submitted with dependencies on expanded main job **22982030** and forward-consistency diagnostic **22982064**. It uses the expanded study's identical retained cohorts and frozen calibration groups. Diagnostic/score failures retain strict guards; no automatic tolerance relaxation. The failed old500-image supplementary attempts remain historical and are not scheduled again.

### Individual L/G/D ablations on the expanded cohort

Added six supplementary one-feature logistic models per classifier: **IG L/G/D alone and Grad-CAM L/G/D alone**. Use original frozen fitting/calibration image groups and identical test cohorts; no maps/inference recomputed. Slurm **22990849** runs on CPUs after main22982030. See [ablation specification](docs/INDIVIDUAL_LGD_ABLATIONS.md). Report all singles alongside combined static/dynamic/interaction models; no test-based replacement of primary methods.

### Overnight completion update (November15)

Expanded primary job22982030 completed in **16h36m01s**; entropy/Energy job22982067 completed in **41m12s**. Evaluation tables and frozen artifacts have been synced locally (large development/test collections and supplementary per-observation scores remain on Rivanna). Individual ablation job22990849 failed before fitting because its code-hash key used a Mac absolute path. The key was corrected to the same source's project-relative path with the identical SHA256, original metadata preserved; all scientific settings unchanged. Replacement ablation job **23004665** was submitted. No ablation outcome should be claimed until it completes.

### All scheduled evaluations complete (November15)

Individual L/G/D ablations job23004665 completed in3m06s for all three architectures. Main expanded study, entropy/Energy supplement and individual ablations are now complete. Individual-ablation outputs are synced under the expanded root's individual_lgd_ablations/. Findings do not establish that all three components together are consistently best: ResNet IG G alone detects15.1% with9.2% false alarms versus IG dynamic14.6%/15.4%; ViT Grad-CAM G alone detects11.8% versus dynamic5.5%, with slightly higher FPR4.4% versus4.0%. Interpret these as supplementary descriptive outcomes; preserve combined primary methods and report all ablations.

### Additional-corruption smoke test (November15)

Authorized frozen-method extension: ResNet-50 on official shot noise, motion blur, fog and JPEG compression, plus saved Gaussian-blur reference. Use a fixed200-candidate class-balanced subset of the expanded test IDs (seed20261117), unchanged expanded-study models/thresholds, IG64 and no fitting. Report all selected corruptions, exclusions, same-within-corruption cohorts and a cross-corruption common-image comparison. Rivanna job **23006811**, oneA100,4h limit. [Panel specification](results/additional_corruption_smoke/README.md). No similarity/encoding methods or cluster/filter thresholds are active in this panel; the completed expanded study remains unchanged.
