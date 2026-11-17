# Final figures and tables

This is the presentation/report package for the completed official CIFAR-10-C study. Main evaluation: three frozen architectures,5000 development/2000 test **candidates**, with valid architecture-specific cohorts in Table 1. Supplementary entropy/Energy and individual-component ablations use the exact corresponding cohorts. The four-corruption panel is a smaller exploratory ResNet-only extension.

## Four figures

### Figure 1 — Frozen warning sensitivity and false alarms

[PNG](figure1_warning_and_false_alarms.png) · [PDF](figure1_warning_and_false_alarms.pdf)

**Caption:** Early-warning sensitivity (top) and trajectory-level false-alarm rate (bottom) on official Gaussian blur for frozen ResNet-50, ViT-B/16 and Swin-T. Separate IG/Grad-CAM L/G/D+trend detectors are compared with MSP, ODIN, entropy and Energy. Thresholds are calibrated using development Gaussian noise/brightness and are not retuned on test. Dotted lines indicate the5% calibration target, not a guaranteed test rate. Cohorts and failure denominators are in Table 1; exact values and all-failure mean severity-step lead are in Table 2. Entropy/Energy are documented supplements.

### Figure 2 — Ranking next-step failures

[PNG](figure2_next_step_discrimination.png) · [PDF](figure2_next_step_discrimination.pdf)

**Caption:** AUROC and average precision for next-step first failure among eligible still-correct nonzero observations. These ranking outcomes use observation-level labels and differ from trajectory timing metrics in Figure 1. Dotted AUROC reference is 0.5; AP reference is the architecture-specific positive prevalence. AP is sklearn average_precision_score, not unspecified trapezoidal PR area. Within-image observations are correlated. No inferential significance is asserted.

### Figure 3 — Attribution-component ablations

[PNG](figure3_attribution_ablations.png) · [PDF](figure3_attribution_ablations.pdf)

**Caption:** Early-warning and false-alarm rates for L alone, G alone, retention-weighted D alone, static L/G/D, L/G/D plus adjacent trends and pairwise-interaction variants. Rows identify classifier/map generator; all methods share within-classifier test cohorts and original development image partitions. The interactions retain6 dynamic inputs plus 15 distinct pairwise products. Singles were added as supplementary ablations; primary methods are not replaced by the best test result. Color scales differ by metric, and all values are percentages.

### Figure 4 — Additional-corruption generalization

[PNG](figure4_corruption_generalization.png) · [PDF](figure4_corruption_generalization.pdf)

**Caption:** Frozen ResNet-50 warning sensitivity against false alarms on official shot noise, motion blur, fog and JPEG compression. All four conditions use the same 187 clean-correct valid images from a fixed200-candidate subset of the expanded test manifest. Titles give corruption-specific failing/nonfailing counts. Upper-left represents a preferable sensitivity/FPR tradeoff. Models, map settings and thresholds are unchanged from expanded development; no outcome-based image selection or recalibration. This exploratory panel is not a three-architecture replication or an untouched-blur confirmation. Source results for all 17 models and Gaussian-blur reference are retained.

## Three tables

| Table | Contents | Formats |
|---|---|---|
| 1 | Candidate, valid development, fitting/calibration, valid test, failure counts and severity 1 failures | [Markdown](table1_cohorts.md), [CSV](table1_cohorts.csv), [LaTeX](table1_cohorts.tex) |
| 2 | Main six-method comparison across three architectures: warning, misses, false alarms, lead, AUROC/AP | [Markdown](table2_main_comparison.md), [CSV](table2_main_comparison.csv), [LaTeX](table2_main_comparison.tex) |
| 3 | Complete individual-component and combined attribution ablations | [Markdown](table3_component_ablations.md), [CSV](table3_component_ablations.csv), [LaTeX](table3_component_ablations.tex) |

`source_all_methods.csv` also preserves confidence trends and all main/supplementary rows. `supplement_corruption_panel.csv` preserves every additional-panel score under both cohort policies; it is a source supplement rather than a fourth principal manuscript table. Percentages and AP are explicitly labeled; no test-based best-method filtering is performed. For a narrow printed layout, Table 2 can be rotated or split by architecture without dropping rows.

## Findings and limits

Attribution-only scores did not demonstrate a consistent advantage over output-based baselines. ResNet IG sometimes offers fewer alarms at lower sensitivity; transformer confidence baselines generally provide stronger sensitivity and ranking. Some components outperform combined attribution models, so combination is not an established universal improvement. Shot-noise Grad-CAM catches two more failures than MSP in the small panel but adds one false alarm; Energy matches its warning count with fewer false alarms. These are descriptive findings, not claims of statistical superiority.

Clean references and ordered histories are required; severity steps are not elapsed time; thresholds need not transfer at5% FPR; checkpoint exposure is not fully known; only specific maps/layers/corruptions are tested. No universal negative claim about attribution methods is justified. The aggregate figures do not replace Week5 qualitative failure examples.

## Rebuild and provenance

```bash
MPLCONFIGDIR=/tmp/asws-final-mpl .venv/bin/python -m scripts.build_final_results
```

No inference, fitting or threshold selection occurs. `provenance.json` records source-file and report-code hashes. Primary source: results/expanded_cifar10c_5000dev_2000test/. Panel source: results/additional_corruption_smoke/. Historical custom-corruption plots are archived separately.
