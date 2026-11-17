# Expanded official CIFAR-10-C sample-size follow-up

The original study completed successfully on one A100 in **1h39m32s** (Slurm22979112). Its 500-clean-correct-development /200-test-candidate results are preserved under results/final_cifar10c/. The additional run is Slurm **22982030**, requesting **48hours**, with a47.5h internal resumable budget. Estimated duration is approximately15–20h by scaling the measured earlier runtime; this is not a guarantee.

## Cohorts

- **5000 development candidates**,500/class: original1000 development-candidate pool plus4000 unused original-test IDs.
- **2000 additional test candidates**,200/class, excluded from recorded warning experiments. Both new pools selected together from6000 unused IDs, label-stratified seed20261116. See reservation/reservation_audit.json.
- Use **all clean-correct candidates** in each architecture, then apply shared attribution-validity exclusions. Actual retained development/test counts are architecture dependent and may be below5000/2000; these numbers are candidate counts, not promises of exact valid sample counts. No class-specific clean-correct quota is forced.
- New test IDs are disjoint from development and prior recorded experiments. Gaussian blur is a previously examined corruption; additional images do not make this untouched-corruption validation or establish classifier training-set exposure.

## Methods and evaluation

Frozen published ResNet50, ViT-B/16 and Swin-T checkpoints. Official CIFAR-10-C Gaussian noise/brightness for development and Gaussian blur for test; severity0–5. IG64, clean-class reference maps, original normalization and L/G/D formulas. Primary IG/Grad-CAM dynamic logistic models retain six L/G/D+adjacent-change inputs. Static/interaction ablations and MSP/ODIN/confidence-trend baselines are unchanged. No Pearson, sparse encoding, cosine, projection or hybrid confidence feature is added to attribution models.

This is a **new larger development fit**, not application of the old500-image weights. Weights and thresholds are refitted using only this run's development fitting/calibration image groups, then frozen before its test evaluation. Original500-image models and results are not overwritten. Target5% calibration false alarms, strict exceedance, common within-classifier cohorts, early warning, missed warnings, lead steps, false alarms, next-step AUROC/AUPRC as before. More observations reduce estimation uncertainty but do not guarantee improved warning performance. In particular more test images are intended to provide more transformer failures for evaluation.

Run `python -m scripts.run_expanded_cifar10c_study run --hours 47.5` to resume. Batch launcher: hpc/expanded_cifar10c_a100.sbatch. Logs: lgd-expanded-22982030.log and this root's hpc_run.log; machine-readable status.json. Collects models sequentially with per-image collection checkpoints.

Entropy/Energy remain separately documented supplementary methods; they are not silently added to this locked attribution protocol. Their original supplementary job failed a prediction-consistency check and must be repaired/evaluated separately before those baseline results are claimed.
