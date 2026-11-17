# Discontinued robust-warning experiments

Group DRO and V-REx are removed from the active project method and future experiment plan. Code and outputs are retained as historical research records, not recommended pipeline entry points.

## Attribution-only ERM / Group DRO pilot

Run `python -m scripts.compare_warning_erm_dro` for a saved-data ResNet-50 development comparison. Proposed scores receive only attribution features/trends from IG, Grad-CAM, or their fusion; confidence predictors belong to a separate baseline. Three convex logistic objectives use identical fitting-only scaling and L2 coefficient 0.01: pooled ERM, equal-group-average ERM, and minimax Group DRO. The latter uses explicit epigraph constraints over corruption-group average logistic losses (analytic gradients, SLSQP). Corruption labels are training group assignments, not predictors. Classifier weights remain fixed.

Results: `results/archive/discontinued/resnet_warning_erm_dro/FINDINGS.md`. Group DRO lowers worst-corruption out-of-fold logistic loss for all three recipes, but only Grad-CAM improves early-warning point estimates (141 vs 132 ERM warnings; 5.6% vs 6.0% false alarms). IG and fusion detect fewer failures under DRO. Better robust training loss does not guarantee better alarm timing or calibration. All 38 tests pass; original frozen/final artifacts are unchanged.

This uses only two observed development corruption families, both present during fitting. It does not establish unseen-corruption generalization. A further domain-generalization study needs several development corruption families with leave-one-corruption-out validation and an unexamined confirmation condition; previously viewed blur outcomes cannot be used for fresh tuning claims.

## Attribution-only V-REx pilot

V-REx was tested on existing ResNet development features using beta=1, the same image folds and calibration as ERM. Grad-CAM detects 137/515 failures early with 4.74% false alarms, versus ERM 132/515 and 5.98%. Detection improvement is uncertain; IG worsens and fusion does not improve. This is observed-corruption development evidence, not unseen-domain confirmation. See [findings](../results/archive/discontinued/resnet_warning_vrex/FINDINGS.md) for protocol, uncertainty and reproduction. All 40 tests pass.

## V-REx fresh-image confirmation

On 475 clean-correct fresh ResNet images under known blur, V-REx warned early for 97/407 failures with 30.88% false alarms, versus ERM 66/407 with 22.06% and MSP 87/407 with 26.47%. V-REx AUROC/AP were 0.689/0.588 versus MSP 0.741/0.660. The development false-alarm benefit did not transfer. See `results/archive/discontinued/fresh_resnet_vrex_confirmation/`. No claim that these methods are universally ineffective follows from this pilot.
