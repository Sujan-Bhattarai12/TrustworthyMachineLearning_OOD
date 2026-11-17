# Individual-component attribution ablations

These supplementary ablations answer whether the combined L/G/D warning model improves on any single component in the **expanded 5000-development-candidate /2000-test-candidate study**. They were added during that run after initial ResNet results, so they are documented as supplementary and cannot be claimed as original predeclared primary methods.

For each of ResNet-50, ViT-B/16 and Swin-T, fit six one-feature warning models:

| Attribution map | Ablations |
|---|---|
| Integrated Gradients | L alone; G alone; D alone |
| Grad-CAM | L alone; G alone; D alone |

D is the existing retention-weighted Wasserstein distance. Each input is the single raw component; no trends, interactions, confidence, Pearson, cosine, projections or encodings are included. A StandardScaler and logistic regression (C=1, max_iter2000, random_state42) match the primary model family. Model coefficients learn the direction from fitting data; the score is not manually flipped after testing.

Use exactly the original frozen fitting/calibration image groups and common-valid development/test trajectories for each architecture. Fit using original eligible next-step-failure observations. Calibrate on nonfailing calibration trajectory maxima at target5% FPR with strict exceedance. Save weights and thresholds before reading that architecture's test observations. Check test cohort keys against original primary evaluation. No attribution maps or classifier inference need to be recomputed.

Report early-warning and missed-warning rates, false alarms, lead steps, next-step AUROC/AUPRC and denominators alongside existing combined static, dynamic and interaction models. All six singles must be reported without choosing a new primary winner from test results. Different achieved FPRs mean higher detection alone does not establish a better tradeoff.

Job **22990849** runs on CPUs after expanded primary job22982030 completes, independently of the entropy/Energy supplementary job. Outputs: `results/expanded_cifar10c_5000dev_2000test/individual_lgd_ablations/`, including per-classifier frozen specifications, warning models, comparison.csv, comparison_with_primary.csv and trajectories.csv; consolidated comparison_all_classifiers.csv. Entry point: `scripts/evaluate_individual_lgd.py`. Slurm launcher: `hpc/individual_lgd_ablations.sbatch`.

This addition does not change the main run's code, fitted weights, locked protocol or results.
