# Final study specification

## Research question and scope

Can changes in gradient-based attribution maps warn of a classifier's first error under increasing corruption, and do development-fitted rules transfer to an excluded corruption better than output-based warning scores? This is **corruption/distribution-shift failure warning**, not detection of previously unseen semantic classes. All images retain CIFAR-10 labels. The three frozen classifiers are ResNet-50, ViT-B/16 and Swin-T. Local published checkpoint configurations, weights, revisions and provenance are under checkpoints/huggingface/. No image-classifier fine-tuning occurs.

The main completed source is results/expanded_cifar10c_5000dev_2000test/. Its locked_protocol.json and per-classifier frozen specifications define actual execution. Only precursor setup records remain at results/final_cifar10c/; superseded results were removed.

## Data and architecture-specific cohorts

Use official CIFAR-10-C arrays, not regenerated corruptions. Severity1–5 selects its corresponding10000-image block; severity 0 uses original clean CIFAR-10 test pixels. Loader validates shapes, type and clean-label ordering. Local file hashes are preserved; the downloaded release archive's MD5 was not independently verified. Published checkpoint exposure outside the recorded experiments is not independently established.

Development has 5,000 candidates (500/class): the previous1000 development pool plus 4,000 additional candidates. Test has 2,000 additional candidates (200/class), disjoint from development and prior recorded warning experiments. Candidate sampling is label-stratified, without selection by warning score. Retain clean-correct images, then exclude whole invalid attribution trajectories equally for every method within architecture.

| Classifier | Valid development images | Fitting | Calibration | Valid test | Failing | Nonfailing |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 4745 | 3558 | 1187 | 1885 | 1352 | 533 |
| ViT-B/16 | 4940 | 3705 | 1235 | 1965 | 110 | 1855 |
| Swin-T | 4799 | 3599 | 1200 | 1920 | 211 | 1709 |

All corruption/severity observations of one image remain in the same development partition. Development uses Gaussian noise and brightness. Test uses Gaussian blur, excluded from detector fitting; related/custom blur and later official blur outcomes have been examined during the broader project. Therefore future tests on that condition cannot be called untouched-corruption confirmation. The displayed5000/2000 are candidate counts, not the valid cohort sizes.

## Maps and attribution metrics

Generate explanations for the fixed clean predicted class at every severity. Clean-correct selection makes that class the true label at baseline, but the target is not switched when the model changes prediction. IG uses a black baseline,64 Gauss–Legendre steps and absolute RGB attribution sum. The main collector calls Captum directly. Raw positive Grad-CAM uses architecture-specific layers and spatial/token reshaping. Both produce32×32 nonnegative maps. Transformer Grad-CAM depends on that adaptation and layer choice.

For clean map A and corrupted map B, use a=A/max(A), b=B/max(A). With epsilon stabilization and p=1:

- L = Σ [a_i/(Σa+epsilon)] max(a_i−b_i,0).
- G = mean[clip(1−a_i/(max(a)+epsilon),0,1) max(b_i−a_i,0)].
- Downsample maps to8×8 using area interpolation, normalize each to unit mass, and calculate spatial optimal-transport cost W using Euclidean ground distances on coordinates in[0,1].
- Retention R = min(1,Σb/(Σa+epsilon)); D = R W.

D is **retention-weighted** distance, not unweighted W. Empty clean maps are invalid. When corrupted attribution vanishes below the implementation cutoff, W is assigned0 by convention while loss/retention record decay; do not interpret0 as proof of unchanged focus. The exact implementation is asws_components in scripts/collect_attribution_metrics.py.

Features at severity s compare with clean. Trends ΔL_s=L_s−L_(s−1), and similarly G/D, describe adjacent changes in those clean-reference quantities. Official severity steps are not equal physical distances or elapsed time.

## Warning models and calibration

Primary: separate IG and Grad-CAM **six-feature** models [L,G,D,ΔL,ΔG,ΔD]. StandardScaler is fitted only on fitting images, followed by LogisticRegression(C=1,max_iter=2000,random_state=42). Logistic regression is a binary classifier. It estimates association with next-step first failure, not a guaranteed calibrated probability under new shifts. Attribution-only inputs exclude confidence, logits, class labels and severity.

Training uses still-correct nonzero observations before first failure, with label1 when the next severity is first failure. Nonfailing trajectories exclude the final horizon from next-step training/ranking. Primary ablations are static L/G/D and all 15 pairwise products of distinct standardized dynamic features, retaining the original 6 (21 inputs total; no squares). Supplementary individual models use L alone, G alone or D alone for each map generator with the same model family, cohorts and partitions. These supplements were added during execution; report them as such and do not select a new primary based on test results.

For every score, take the maximum over severities1–5 on **nonfailing calibration trajectories**. Choose a threshold giving at most 5% empirical calibration trajectory alarms, with strict score>threshold. This is a calibration operating point, not guaranteed test control. Save fitted weights, preprocessing and thresholds before evaluating test observations. Logistic model fitting uses both labels; only this threshold calibration uses nonfailing trajectories. Linear regression fitted to only zero/nonfailure labels would learn a constant target and is not part of the project.

## Output-based baselines

- **MSP:** 1−max softmax(logits); Hendrycks and Gimpel's confidence baseline.
- **ODIN:** 1−max softmax after temperature 1000 and a gradient-based confidence-increasing input step epsilon 0.0014. The implementation differentiates through preprocessing, perturbs raw[0,1] pixels and clips them. This bounded-pixel adaptation and fixed defaults are documented; no test tuning or retraining is done.
- **Predictive entropy:** −Σp ln p, deterministic softmax at temperature 1. Shannon entropy applied to class probabilities, not MC Dropout.
- **Energy:** −T logΣexp(logits/T), T=1, following Liu et al.'s post-hoc scoring formula without energy-based retraining. Higher energy is the warning direction.
- **Confidence trends:** separate learned control using unconfidence and its adjacent change; never combined into attribution-only inputs.

Entropy/Energy are supplementary additions scored on the exact expanded cohorts with original calibration groups. Their inference path preserves the collector's single-image loader tensor layout and verifies predictions/MSP. Failed earlier implementations are not evidence. MC Dropout is excluded because the selected unchanged checkpoints lacked suitable active trained dropout.

## Additional-corruption extension

ResNet-50 only: select200 class-balanced candidates from expanded test IDs (seed20261117), retain187 valid images. Freeze all existing warning models/thresholds. Collect shot noise, motion blur, fog and JPEG compression; reuse Gaussian blur as a reference. All187 images are valid in all five conditions, enabling paired image comparisons. Different conditions have different failure/nonfailure counts. No corruptions are added to fitting. This is an exploratory small panel, not an independent three-architecture replication.

## Evaluation and interpretation

Primary outcomes are trajectory-level early-warning sensitivity, missed warnings, any-alarm FPR on nonfailing trajectories and lead in severity steps (misses0). Alarms at/after first error are not early. Failures at severity 1 have no nonzero pre-error opportunity and count as misses. Secondary AUROC and average precision rank next-step failure among eligible still-correct observations; observations within one image are correlated. Average precision is sklearn's AP, not an unspecified trapezoidal PR area. No inferential significance or universal state-of-the-art claims are made. A false alarm means no error through severity 5, not safety under every future shift.

The completed data do not demonstrate consistent attribution superiority. Component ablations show that combining L/G/D is not always beneficial. Confidence generally ranks failures more strongly, and calibrated thresholds can transfer poorly. Conclusions apply to these checkpoints, maps, references, corruption families and sample sizes. Clean-reference access, ordered histories, magnitude normalization, attribution faithfulness, model-specific maps and external checkpoint exposure constrain generalization. No causal claim that the maps reveal the model's reasoning is required.

## Method references

- [Hendrycks and Gimpel,2017: MSP](https://arxiv.org/abs/1610.02136).
- [Liang, Li and Srikant,2018: ODIN](https://openreview.net/pdf?id=H1VGkIxRZ).
- [Liu et al.,2020: Energy-based OOD Detection](https://arxiv.org/abs/2010.03759).
- [Shannon,1948: entropy](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x). The information measure, not an attribution/failure detector proposal.
- [Sundararajan, Taly and Yan,2017: Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html).
- [Selvaraju et al.,2017: Grad-CAM](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html).
- [Hendrycks and Dietterich,2019: corruption robustness](https://arxiv.org/abs/1903.12261); [official CIFAR-10-C release](https://zenodo.org/records/2535967).
