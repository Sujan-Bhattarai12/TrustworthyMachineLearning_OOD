# Supplementary entropy and Energy baselines

These two deterministic, forward-only scores supplement MSP, ODIN and confidence trends. They do not change the original locked attribution protocol, classifier weights, attribution features or primary results. They were added after the primary run started and must be identified as supplementary comparisons.

For logits z and p = softmax(z), predictive entropy is H = −Σ p_k ln(p_k). Higher entropy indicates greater uncertainty. I use natural logarithms and temperature 1. This is Shannon entropy applied to the classifier's predictive distribution, not MC Dropout or an ensemble uncertainty estimate. The foundational reference is Claude E. Shannon (1948), **A Mathematical Theory of Communication**, Bell System Technical Journal 27(3), 379–423. [Paper and DOI](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x). Shannon did not propose this project's failure detector.

Energy is E = −T ln Σ exp(z_k/T), with T fixed to 1. Higher energy triggers a warning. I follow the post-hoc energy scoring formulation of Weitang Liu, Xiaoyun Wang, John D. Owens and Yixuan Li (2020), **Energy-based Out-of-distribution Detection**, NeurIPS 33. [Paper](https://arxiv.org/abs/2010.03759). I use the score on frozen classifiers, without the paper's optional energy-based training. Its original OOD application does not guarantee early-warning performance under CIFAR-10-C corruption.

Both use the same official images, severity levels 0–5, common-valid cohorts and original frozen development/calibration image split. Thresholds are calibrated on nonfailing development calibration trajectory maxima at a target 5% false-alarm rate, using strict exceedance. Thresholds are saved before supplementary test scoring; temperature and score direction are not tuned on test outcomes. The 5% calibration target does not guarantee a 5% test false-alarm rate.

Implementation: `src/supplementary_scores.py` and `scripts/evaluate_supplementary_baselines.py`. Outputs, logits, frozen thresholds and protocol hashes go to `results/final_cifar10c/supplementary_entropy_energy/`. Each classifier has `comparison_with_primary.csv`, containing original methods and supplementary rows. Main-study files are never overwritten. Forward predictions must match the original collection before scores are accepted.

After primary collection and evaluation finish:

```bash
python -m scripts.evaluate_supplementary_baselines setup
python -m scripts.evaluate_supplementary_baselines run
```

Run setup once; it refuses to overwrite the supplementary protocol. GPU execution is recommended, but no IG or Grad-CAM calculation is repeated.

## Prediction-consistency repair

The initial supplementary job22980164 failed before producing accepted baseline results: batched inference did not match the locked collector's predictions/MSP within its guard. The original protocol and source remain unchanged, including their code hashes. A separate `scripts/evaluate_supplementary_baselines_v2.py` uses **single-image `torch.no_grad()`** forwards, matching the original collector's execution, with the same label/MSP checks and tolerances. It writes to `results/final_cifar10c/supplementary_entropy_energy_v2/` with a new supplementary protocol; temperatures, formulas, cohorts and calibration rules are unchanged. Slurm22982048 was submitted for validation and evaluation. Its consistency check must pass before any scores are accepted; submission alone does not establish that the mismatch is resolved.

The expanded5000/2000 L/G/D study is independent. Neither its locked source nor its weights/thresholds are modified by this repair.

## Expanded-cohort supplement replaces the 500-image supplement

At the user's request, entropy/Energy evaluation now targets **only the expanded 5000-development-candidate /2000-test-candidate study** under `results/expanded_cifar10c_5000dev_2000test/`. It uses the exact architecture-specific common-valid cohorts and frozen calibration image groups from that study. No further old500-image baseline rerun is scheduled. Old failed attempts are historical records.

Entry point: `scripts/evaluate_expanded_supplementary_baselines.py`; launcher: `hpc/expanded_supplementary_baselines.sbatch`. Protocol/output folder: the expanded root's `supplementary_entropy_energy/`. Temperatures remain1 and thresholds remain development-calibrated at target5%; no test tuning. The forward path preserves the original loader tensor's memory layout instead of repacking it. Original strict label/MSP consistency tolerances are retained.

A separate small GPU diagnostic verifies forward consistency against saved observations, without fitting or evaluating warning scores. The full supplementary job depends on successful completion of both that diagnostic and expanded primary job22982030. A failed diagnostic blocks automatic baseline scoring; it does not relax the consistency requirement. New job IDs are documented in the active project plan.
