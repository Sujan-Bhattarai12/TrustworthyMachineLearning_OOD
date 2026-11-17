# Attribution-Based Failure Detection Under Distribution Shift

This completed project asks whether changes in attribution maps warn of image-classification failure earlier or more reliably than output-based confidence scores.

**Main finding:** the evaluated attribution-only warning methods did not demonstrate a consistent advantage over confidence baselines under official CIFAR-10-C corruption shift. Some component ablations and individual corruption cases show different tradeoffs; they do not establish universal superiority or universal failure of attribution methods.

Start with the [final figures and tables](results/reporting/final/README.md), [study specification](docs/FINAL_STUDY.md), and [script reading guide](docs/CODE_GUIDE.md).

## GitHub contents

This repository contains code, configuration, and study documentation. Datasets, model weights, split CSVs, generated experiment data, figures, and the presentation binary remain local and are excluded by `.gitignore`. Links to those artifacts describe the full local project; rebuilding reports requires the saved evaluation files, which are not included here.

## Final study

- Frozen published CIFAR-10 **ResNet-50, ViT-B/16 and Swin-T** checkpoints.
- Official pre-generated **CIFAR-10-C**, severity 0–5;0 is the corresponding clean image.
- **5000 development candidates**, Gaussian noise and brightness; **2000 additional test candidates**, Gaussian blur. Actual clean-correct/common-valid image counts vary by architecture.
- **Integrated Gradients (64 steps)** and **Grad-CAM** generate maps for the fixed clean predicted class.
- Attribution Loss **L**, Attribution Gain **G**, retention-weighted Wasserstein distance **D** and their adjacent-severity changes feed separate attribution-only logistic classifiers.
- Primary IG/Grad-CAM L/G/D+trend models; combined static and pairwise-interaction ablations; supplementary **L alone, G alone, D alone** for both map generators.
- Baselines: **MSP, ODIN, predictive entropy, Energy**, with confidence trends as an additional control. Entropy/Energy and individual components were added as documented supplements.
- Models and preprocessing fitted on development fitting groups; thresholds calibrated on separate image groups, targeting5% nonfailing-trajectory alarms, then frozen before test evaluation. Achieved test false alarms can differ.
- A supplementary frozen **ResNet-50** panel evaluates shot noise, motion blur, fog and JPEG compression on the same 187 valid images. It is an exploratory extension, not another three-architecture benchmark.

Logistic regression is the secondary **classification** model; the image classifiers are not fine-tuned. No confidence feature enters attribution-only warning models. Pearson/cosine, sparse encodings, projections, clustering and alarm-filter experiments are archived and excluded from the final method.

## Results to use

| Location | Role |
|---|---|
| [results/reporting/final/](results/reporting/final/README.md) | Four figures, three tables, captions and source hashes |
| `results/expanded_cifar10c_5000dev_2000test/` | Authoritative expanded data, maps, frozen models, evaluations and supplements |
| [results/additional_corruption_smoke/](results/additional_corruption_smoke/FINDINGS.md) | Additional-corruption panel and complete outcomes |
| `results/final_cifar10c/` | Only precursor development manifest and locked protocol retained for expanded setup; superseded outputs removed |
| `results/validation/`, `figures/archive/historical_validation/` | Historical attribution-validation evidence; provenance limits apply |

## Rebuild the final report

```bash
MPLCONFIGDIR=/tmp/asws-final-mpl .venv/bin/python -m scripts.build_final_results
```

This reads saved evaluation files; it does not rerun classifiers, fit models or tune thresholds. Each figure is PNG/PDF and each table is CSV/Markdown/LaTeX. See [metrics](docs/METRICS_GUIDE.md) and [presentation order](docs/PRESENTATION_GUIDE.md).

See [the documentation index](docs/README.md) for the final guides. [Code synchronization](docs/CODE_SYNC.md) distinguishes the cleaned local tree from the earlier Rivanna version. Historical sources and records are archived; restoring their original paths/bytes is necessary before guarded historical reruns. Completed results are unchanged.
