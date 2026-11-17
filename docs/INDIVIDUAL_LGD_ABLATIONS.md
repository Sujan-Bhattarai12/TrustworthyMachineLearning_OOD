# Completed individual attribution-component ablations

For each architecture and each map generator (IG, Grad-CAM), fit **L alone, G alone and D alone**: six one-feature logistic classifiers per architecture. D is retention-weighted spatial Wasserstein. No trends, interactions, confidence or map-similarity features enter these individual models.

Use the exact primary common-valid expanded cohorts and original frozen fitting/calibration image groups. StandardScaler + logistic regression C=1, max_iter2000, seed42; eligible next-step first-failure labels. Thresholds come from nonfailing calibration trajectory maxima, target 5%, strict exceedance, and are frozen before test observations are loaded. Test cohort keys are verified against the primary evaluation. These are supplementary additions after initial ResNet outcomes, not the original primary protocol or a test-selected replacement.

Implementation: scripts/evaluate_individual_lgd.py. Completed outputs: results/expanded_cifar10c_5000dev_2000test/individual_lgd_ablations/. [Final Table 3](../results/reporting/final/table3_component_ablations.md) presents these alongside combined static/dynamic/interaction models. Gains differ across components and architectures; there is no established universally best combination.
