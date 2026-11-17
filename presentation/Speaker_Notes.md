## 1. Attribution-Based Failure Detection Under Distribution Shift

Suggested time: 30 seconds.

My project asks whether changes in explanations can warn us that an image classifier is approaching a mistake. I compare two attribution methods against four output-based scores on three frozen CIFAR-10 classifiers. The focus is warning before the first error as official corruption severity increases. I will explain the metrics, the frozen evaluation protocol and the trade-off between detecting failures and raising false alarms.

Sources: docs/FINAL_STUDY.md

## 2. Can attribution changes warn before the first error?

Suggested time: 60 seconds.

Confidence tells us how strongly a classifier favors its answer. Attribution maps provide a different view: which input regions influence a selected class score. My hypothesis was that these maps might change before confidence becomes a useful warning. The example here is illustrative. If the first incorrect prediction occurs at severity four and the alarm occurs at severity two, the lead is two severity steps. An alarm at severity four is too late to count as early. This is a corruption-shift experiment with the same ten labels, not a new-class OOD detector.

Sources: docs/FINAL_STUDY.md

## 3. Official CIFAR-10-C and three frozen classifiers

Suggested time: 70 seconds.

I used official pre-generated CIFAR-10-C arrays. Severity zero is the clean image; levels one through five are the published corruption blocks. These remain algorithmic corruptions, not natural distribution shift. Five thousand development candidates were split by image into fitting and calibration groups. Gaussian noise and brightness were used in development. The separate two thousand test candidates were evaluated with Gaussian blur, excluded from fitting. The table gives actual valid image counts after clean-correct and attribution-valid selection. Every method within an architecture uses the same cohort. The base classifiers were not fine-tuned. ViT-B/16 means Vision Transformer Base with sixteen-pixel patches; Swin-T means Shifted Window Transformer Tiny.

Sources: docs/FINAL_STUDY.md; results/reporting/final/table1_cohorts.csv; https://zenodo.org/records/2535967

## 4. IG and Grad-CAM generate the attribution maps

Suggested time: 60 seconds.

Integrated Gradients and Grad-CAM are the two ways I generate explanations. They are not the competing failure-detection baselines. IG integrates gradients along the path from a black baseline to the image, using sixty-four Gauss-Legendre steps. I sum absolute RGB attributions into a nonnegative spatial map. Grad-CAM combines gradients with layer activations and requires architecture-specific spatial or token reshaping. Both maps become thirty-two by thirty-two. At every corruption level I explain the fixed clean predicted class, rather than switching to a new predicted class. The metrics therefore track what happens to evidence for the original decision.

Sources: docs/FINAL_STUDY.md; https://proceedings.mlr.press/v70/sundararajan17a.html; https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html

## 5. Three metrics summarize changes from the clean map

Suggested time: 80 seconds.

Let A be the clean attribution map and B the corrupted map. Both are divided by the clean maximum, so changes include magnitude changes as well as spatial changes. Attribution loss emphasizes decreases at locations that were important in the clean map. Attribution gain emphasizes increases at locations that had low clean attribution. For distance, I area-downsample to eight by eight, normalize to unit mass and compute spatial Wasserstein transport. The final D is multiplied by retained attribution mass, so it is retention-weighted distance, not raw transport cost. A map losing almost all mass can have a small D. Loss and retention capture that decay; small D alone does not mean unchanged attention.

Sources: docs/FINAL_STUDY.md

## 6. A logistic classifier turns the metrics into a warning score

Suggested time: 80 seconds.

For each architecture I fit separate IG and Grad-CAM warning classifiers. The six primary features are loss, gain, distance and their changes from the previous severity. Standardization is fitted only on fitting images. The logistic classifier learns from still-correct observations, labeled one if the next severity produces the first failure. It uses both positive and negative training labels. No confidence, logits, class labels or severity values enter the attribution detector. On nonfailing calibration trajectories I take the maximum score across levels and select a threshold allowing at most five percent empirical trajectory alarms. I save preprocessing, learned coefficients and thresholds before test evaluation. This five percent is a calibration target, not a promise on a new corruption.

Sources: docs/FINAL_STUDY.md

## 7. Four output-based baselines provide the comparison

Suggested time: 55 seconds.

MSP uses one minus the largest softmax probability. Predictive entropy uses the full class probability distribution. Energy uses the log-sum-exp of logits, following the post-hoc scoring formula without energy training. ODIN combines temperature scaling with an input perturbation intended to increase confidence. I use temperature one thousand and epsilon point zero zero one four in a bounded-pixel implementation. All scores are oriented so larger means more warning and use the same nonfailing calibration procedure. Entropy and energy were supplementary additions evaluated on the same expanded cohorts. MC Dropout was excluded because the unchanged checkpoints did not provide suitable trained dropout.

Sources: docs/SUPPLEMENTARY_BASELINES.md; https://arxiv.org/abs/1610.02136; https://openreview.net/pdf?id=H1VGkIxRZ; https://arxiv.org/abs/2010.03759; https://doi.org/10.1002/j.1538-7305.1948.tb01338.x

## 8. Early warnings and false alarms answer different questions

Suggested time: 65 seconds.

The primary evaluation unit is an image trajectory. Early-warning rate divides trajectories warned strictly before their first error by all trajectories that fail. False-alarm rate uses a different denominator: trajectories that remain correct through severity five. Any alarm on such a trajectory is a false alarm. Lead is measured in discrete severity steps, and missed failures contribute zero to the all-failure mean. Failures already at level one have no nonzero pre-error opportunity and count as misses. AUROC and average precision are secondary ranking metrics on eligible still-correct observations labeled for next-step failure. They do not directly measure how early an alarm occurs. AUPR in these files is sklearn average precision, not trapezoidal integration.

Sources: docs/METRICS_GUIDE.md

## 9. Confidence scores detected more failures early

Suggested time: 80 seconds.

These are the main Gaussian-blur test results. Higher early-warning rate is better, but this chart must be read with the false-alarm chart next. On ResNet, energy warns on twenty-five percent of failing trajectories, MSP on twenty-one point four percent, Grad-CAM on sixteen point nine and IG on fourteen point six. On ViT, MSP reaches twenty-seven point three percent compared with ten point nine for IG and five point five for Grad-CAM. On Swin, MSP reaches twenty-four point six percent while the attribution methods are below two point four percent. Energy has the highest ResNet sensitivity; MSP is the strongest overall early-warning comparator across architectures. These are descriptive comparisons, not significance claims.

Sources: results/reporting/final/table2_main_comparison.csv

## 10. The false-alarm cost varies sharply across architectures

Suggested time: 70 seconds.

Here lower is better. ResNet is the clearest example of threshold transfer problems: although calibration targeted five percent, test false alarms range from roughly fifteen to twenty-five percent. IG is more conservative than MSP there but also warns on fewer failing trajectories. For ViT and Swin, MSP combines higher early-warning sensitivity with lower false-alarm rates than either primary attribution model. ODIN is extremely conservative on those models, which explains its low warning rate. We cannot judge a method by sensitivity alone or assume the development threshold guarantees the same test error rate. The denominator is nonfailing trajectories, so these percentages should not be subtracted from warning rates as if they represented the same population.

Sources: results/reporting/final/table2_main_comparison.csv

## 11. Ranking metrics tell a similar story

Suggested time: 70 seconds.

These two charts evaluate next-step failure ranking rather than alarm timing. On ResNet, entropy and MSP have AUROC around point eight eight seven, ahead of the attribution models. For ViT and Swin, confidence-based ranking is especially strong, while the attribution warning scores are weaker. Average precision is sensitive to failure prevalence, so comparisons within an architecture and cohort are the most useful. The primary attribution models do contain some predictive information, particularly on ResNet and ViT, but predictive association is not enough to establish a better warning system. The results do not support consistent superiority over output-based scores.

Sources: results/reporting/final/table2_main_comparison.csv

## 12. Combining the attribution metrics was not always best

Suggested time: 65 seconds.

I also tested static loss, gain and distance, the primary trend model, and interaction terms. Supplementary single-component models fit loss alone, gain alone or distance alone with the same classifier family and partitions. These examples show why it would be misleading to say all three metrics together are always best. ResNet IG gain alone has similar early-warning sensitivity but fewer false alarms than the primary IG model. ViT Grad-CAM gain alone has higher sensitivity with a small false-alarm increase. These are supplementary observations after earlier outcomes were available. I keep the primary model unchanged instead of selecting whichever test result looks best. Interactions and trends did not produce a consistent advantage across all models.

Sources: results/reporting/final/table3_component_ablations.csv

## 13. Additional corruptions did not show a consistent advantage

Suggested time: 65 seconds.

For an exploratory extension I retained the frozen ResNet warning models and thresholds and evaluated shot noise, motion blur, fog and JPEG compression. The same one hundred eighty-seven valid images appear in every condition, although failure counts differ by corruption. To keep the chart readable I show Grad-CAM, MSP and energy; the complete results also include IG, ODIN and entropy. Grad-CAM slightly exceeds MSP sensitivity on shot noise but has more false alarms. Energy matches Grad-CAM there with fewer false alarms. Across the remaining corruptions the attribution model does not show a consistent benefit. This is a small, ResNet-only exploratory panel after prior blur analysis, not an independent replication across all three architectures.

Sources: results/reporting/final/supplement_corruption_panel.csv; results/additional_corruption_smoke/FINDINGS.md

## 14. Attribution scores did not consistently beat confidence

Suggested time: 50 seconds.

The central finding is that attribution changes can be predictive, but the tested attribution-only warning rules did not consistently outperform output-based baselines. MSP is the strongest overall comparator here, with energy leading ResNet early-warning sensitivity. This is a useful empirical result rather than evidence that attribution can never work. The scope is limited to these checkpoints, fixed-class maps, clean references, official corruption levels and detector designs. Severity is not elapsed time, the test corruption has been examined during this broader project, and repeated observations of the same image are correlated. The reproducible contribution is the attribution metrics and controlled evaluation showing where predictive association fails to translate into reliable early warning.

Sources: docs/FINAL_STUDY.md

## 15. Backup: exact attribution definitions

Backup slide, outside the 15-minute talk.

These are the exact definitions in the final collector. Epsilon stabilizes division. Maps are normalized with the clean maximum. W is optimal transport on unit-mass eight by eight maps with Euclidean ground distances in the unit square. When corrupted mass vanishes, W equals zero by convention, while loss and retention describe the decay.

Sources: docs/FINAL_STUDY.md

## 16. References

Backup slide, outside the 15-minute talk.

Selected method references. Full provenance, specifications and result tables are included in the project documentation.

Sources: docs/FINAL_STUDY.md; docs/SUPPLEMENTARY_BASELINES.md
