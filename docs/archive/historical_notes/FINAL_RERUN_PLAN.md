> Superseded by [the official CIFAR-10-C protocol](../CIFAR10C_PROTOCOL.md). Do not resume the custom-corruption run described below.

# Final rerun preparation

Status: configuration locked for 500 development images and 200 fresh test candidates per classifier, 64 IG steps, L/G/D + trends as primary (IG and Grad-CAM separately), static and interaction scores as predeclared ablations. Run root: results/final_project_run. Budget: 10 hours per invocation, resumable at image boundaries.

## Agreed scope

- Frozen published CIFAR-10 ResNet-50, ViT-B/16 and Swin-T, original checkpoint preprocessing.
- IG and Grad-CAM explaining the clean predicted class; original L/G/D formula and normalization.
- Gaussian noise and brightness for development, blur for final evaluation; severity 0–5.
- A fresh computation of the study is not automatically fresh data. Previously examined blur conditions must remain disclosed. Record exact reused/new image IDs and any overlap.
- Main baselines: MSP, ODIN and separate confidence trends when comparing dynamic attribution.
- Same image identities across feature variants within each classifier, fitting/calibration/evaluation separated, fixed 5% nonfailing-trajectory calibration target.
- Report all predeclared primary methods and ablations, including misses and false alarms; do not select a final winner using test outcomes.
- Keep static scores, dynamic logistic scores and interactions explicitly identified. No confidence/attribution hybrids, DRO, V-REx or new shift datasets.

## Naming

- `scripts/collect_attribution_metrics.py`: image predictions and IG/Grad-CAM L/G/D collection (formerly run_asws_experiment).
- `scripts/evaluate_warning_models.py`: warning comparison (formerly fit_aws_logreg).
- `scripts/run_benchmark_suite.py`: generic classifier collection orchestration (formerly run_classifier_benchmarks).

`scripts/run_final_project.py` is the locked end-to-end orchestrator. It saves logistic pipelines, human-readable coefficients/scalers, source hashes and calibrated thresholds before test evaluation. The generic suite is not the final protocol.

## Checks and outputs to complete

1. Verify regression tests, source syntax, checkpoint hashes and original manifests.
2. Lock primary method, IG steps, image counts, baselines, ablations, seeds and output directory.
3. Validate a tiny end-to-end collection/fit/freeze/evaluation before expensive collection.
4. Collect new development observations; fit/calibrate source-only frozen scores; collect final evaluation observations and evaluate without retuning.
5. Generate one consistent result table and presentation figures with exact counts, score definitions and run provenance.
6. Summarize unsupported exploratory findings in an archive note. Retain original evaluation records and test exposure manifests rather than deleting negative evidence.

No historical result or frozen JSON is rewritten to reflect the new names or IG settings. Archives are preferable to irreversible deletion: the repository currently reports project files as untracked, so git is not a recovery mechanism for removed files.
