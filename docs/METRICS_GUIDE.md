# Metrics used in the final results

There are three different layers: **attribution features describe maps**, **warning scores estimate risk**, and **evaluation metrics judge the warning system**. They should not be used interchangeably.

| Quantity | Meaning | Important qualification |
|---|---|---|
| Classification correctness | Predicted label equals ground truth | Labels are used for research evaluation, not warning-model inputs |
| Classification cross-entropy | −log probability of the true class | Different from Attribution Loss; not available without the true label |
| L: Attribution Loss | Attribution decreases at clean-important pixels | High L does not itself prove impending failure |
| G: Attribution Gain | Attribution increases at clean-low-importance pixels | New attribution can be useful or harmful; gain need not increase with severity |
| W: spatial Wasserstein | Cost of moving normalized map mass spatially | Uses8×8 distributions and normalized spatial coordinates |
| D = retention × W | Spatial transport weighted by remaining attribution mass | Can decline when attribution collapses, despite stronger corruption |
| ΔL, ΔG, ΔD | Difference in clean-reference metrics between adjacent levels | Severity steps, not derivatives in real time |
| Warning score | Logistic output from attribution features, or a baseline score | A ranking/risk score; not automatically a calibrated probability under shift |
| Threshold | Score cutoff for an alarm | Chosen on development calibration groups, never tuned on final test outcomes |

For exact feature formulas and zero-map conventions, see [study specification](FINAL_STUDY.md) and asws_components in the collector.

## Trajectory-level warning outcomes

A trajectory is one original image under one corruption at severities 0–5. The **first failure** is the first incorrect classification, even if a later prediction recovers. No warning is counted at clean severity 0.

- **Early-warning rate:** failing trajectories with an alarm strictly before first error / all failing trajectories. An alarm at the error is too late. Failures at severity 1 have no nonzero pre-error opportunity and count as misses.
- **Missed-warning rate:** failing trajectories without an earlier alarm / all failing trajectories; equals1−early-warning rate.
- **False-alarm rate:** nonfailing trajectories with any alarm at severities1–5 / all nonfailing trajectories. Several alarms within one trajectory count once. “Nonfailing” means correct through severity 5, not guaranteed safe beyond that horizon.
- **Lead:** first-error severity minus first-alarm severity when early. Primary mean lead averages across **all failing trajectories**, assigning 0 to misses. This avoids rewarding a method for warning early on only a few cases. Units are severity steps, not seconds.

Early-warning and false-alarm rates use different denominators. Always inspect the associated counts. A higher warning rate alone is not a better system; more alarms often increase both sensitivity and false alarms. The final plots show them together.

## What the5% calibration target means

The score model is fitted on both next-step-positive and negative fitting observations. Separately, threshold calibration takes the maximum score over each **nonfailing calibration trajectory** and sets a strict-exceedance cutoff with at most 5% empirical alarms on those trajectories. This does not guarantee 5% test FPR when distributions change. It also is not5% of all severity observations, not5% misclassification probability and not a0.5 probability cutoff.

## Next-step ranking outcomes

AUROC and average precision use eligible **still-correct observations**, labeled positive if the next severity is the first error. Severity0, post-error observations and the final nonfailing horizon are excluded. These differ from the trajectory-level timing denominators above.

- **AUROC:** how well scores rank a randomly drawn positive above a negative;0.5 is non-discriminative ranking,1 is perfect. Strong AUROC does not guarantee an effective fixed threshold or long lead.
- **Average precision (AP):** sklearn average_precision_score, reported as aupr in raw CSVs. It summarizes precision/recall; it is not necessarily identical to trapezoidal area under a PR curve. Its uninformative population reference is positive prevalence, so inspect prevalence before comparing architectures.
- Observations from one image and related corruptions are correlated. The report provides descriptive estimates, not claims of independent observation counts or statistical significance.

A warning-rate/FPR ratio can be descriptive but is unstable when false-alarm counts are small and undefined at zero. It does not replace sensitivity, costs, counts or lead. The final study does not optimize or select a primary model using that ratio on test data.

## Reading the four figures

1. Warning/false alarms: frozen operating points; the dotted5% line is a development target only.
2. AUROC/AP: next-step ranking, not warning timing or probability calibration. AP dotted lines show each architecture's next-step prevalence.
3. Ablations: compares individual components, combined components, trends and interactions. All variants are reported rather than selecting the highest test number.
4. Corruption panel: upper-left is preferable (more warnings of failures, fewer false alarms); the same 187 images are used, but failure/nonfailure counts differ by corruption.
