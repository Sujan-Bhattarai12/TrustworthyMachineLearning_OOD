# Additional-corruption smoke-test findings

ResNet-50, frozen expanded-study warning models and thresholds. Job23006811 completed in10m41s. The200-candidate subset retained187 clean-correct valid images for each corruption; all187 are also in the cross-corruption intersection (see cohort.json). No model or threshold was tuned on these outcomes. These are exploratory small-cohort results, not proof across all corruption shifts or architectures.

| Corruption | Model | Early warnings | False alarms | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|
| fog | IG dynamic | 1/52 (1.9%) | 52/135 (38.5%) | 0.748 | 0.207 |
| fog | GRADCAM dynamic | 2/52 (3.8%) | 15/135 (11.1%) | 0.751 | 0.236 |
| fog | MSP | 11/52 (21.2%) | 12/135 (8.9%) | 0.882 | 0.405 |
| fog | ODIN | 2/52 (3.8%) | 0/135 (0.0%) | 0.658 | 0.169 |
| fog | Entropy | 9/52 (17.3%) | 13/135 (9.6%) | 0.882 | 0.405 |
| fog | Energy | 11/52 (21.2%) | 13/135 (9.6%) | 0.841 | 0.359 |
| gaussian_blur | IG dynamic | 23/134 (17.2%) | 7/53 (13.2%) | 0.696 | 0.476 |
| gaussian_blur | GRADCAM dynamic | 22/134 (16.4%) | 12/53 (22.6%) | 0.758 | 0.493 |
| gaussian_blur | MSP | 28/134 (20.9%) | 11/53 (20.8%) | 0.876 | 0.699 |
| gaussian_blur | ODIN | 13/134 (9.7%) | 11/53 (20.8%) | 0.712 | 0.499 |
| gaussian_blur | Entropy | 27/134 (20.1%) | 11/53 (20.8%) | 0.875 | 0.696 |
| gaussian_blur | Energy | 28/134 (20.9%) | 13/53 (24.5%) | 0.831 | 0.649 |
| jpeg_compression | IG dynamic | 7/62 (11.3%) | 29/125 (23.2%) | 0.551 | 0.101 |
| jpeg_compression | GRADCAM dynamic | 12/62 (19.4%) | 23/125 (18.4%) | 0.704 | 0.218 |
| jpeg_compression | MSP | 13/62 (21.0%) | 16/125 (12.8%) | 0.833 | 0.301 |
| jpeg_compression | ODIN | 7/62 (11.3%) | 11/125 (8.8%) | 0.696 | 0.225 |
| jpeg_compression | Entropy | 12/62 (19.4%) | 18/125 (14.4%) | 0.833 | 0.321 |
| jpeg_compression | Energy | 17/62 (27.4%) | 23/125 (18.4%) | 0.821 | 0.304 |
| motion_blur | IG dynamic | 10/78 (12.8%) | 32/109 (29.4%) | 0.579 | 0.138 |
| motion_blur | GRADCAM dynamic | 11/78 (14.1%) | 18/109 (16.5%) | 0.644 | 0.174 |
| motion_blur | MSP | 16/78 (20.5%) | 12/109 (11.0%) | 0.793 | 0.339 |
| motion_blur | ODIN | 5/78 (6.4%) | 3/109 (2.8%) | 0.691 | 0.278 |
| motion_blur | Entropy | 16/78 (20.5%) | 10/109 (9.2%) | 0.794 | 0.364 |
| motion_blur | Energy | 17/78 (21.8%) | 15/109 (13.8%) | 0.779 | 0.383 |
| shot_noise | IG dynamic | 15/149 (10.1%) | 6/38 (15.8%) | 0.617 | 0.403 |
| shot_noise | GRADCAM dynamic | 38/149 (25.5%) | 7/38 (18.4%) | 0.766 | 0.542 |
| shot_noise | MSP | 36/149 (24.2%) | 6/38 (15.8%) | 0.851 | 0.681 |
| shot_noise | ODIN | 42/149 (28.2%) | 6/38 (15.8%) | 0.779 | 0.590 |
| shot_noise | Entropy | 35/149 (23.5%) | 6/38 (15.8%) | 0.850 | 0.679 |
| shot_noise | Energy | 38/149 (25.5%) | 4/38 (10.5%) | 0.837 | 0.658 |

Grad-CAM dynamic warns on38/149 shot-noise failures versus MSP36/149, but has7/38 false alarms versus6/38. Energy matches38 early warnings with only4 false alarms, while ODIN warns on42 failures with6 false alarms. Thus this small sensitivity improvement over MSP does not establish a better overall tradeoff. For fog, motion blur and JPEG, MSP catches more failures with fewer false alarms than either primary attribution model. The panel does not establish a consistent attribution advantage.

comparison.csv preserves all primary/static/interaction, individual L/G/D and confidence-baseline results under both cohort policies. Report all selected types; do not select a new primary variant using the panel. Maps and the large collection CSV remain on Rivanna.
