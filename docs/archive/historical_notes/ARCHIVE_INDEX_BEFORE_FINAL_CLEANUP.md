# Archive index

The [presentation guide](../PRESENTATION_GUIDE.md) identifies retained evidence. Exploratory work is preserved for reproducibility, outside the main presentation workflow.

| Folder | Contents |
|---|---|
| `scripts/archive/` | Experimental fusion, fresh cohorts, robustness objectives, dense grids, diagnostics and older smoke scripts |
| `tests/archive/` | Regression tests for archived experiments, included in full test discovery |
| `results/archive/exploratory/` | Feature diagnosis, interactions, fusion, corruption transfer and map robustness |
| `results/archive/discontinued/` | DRO/V-REx results |
| `results/archive/historical_hybrids/` | Hybrid attribution/confidence studies |
| `results/archive/pilots/` | Early/smoke/superseded collections |
| `results/archive/logs/` | Prior main collection logs |
| `figures/archive/` | Older individual images/illustrations |
| `docs/archive/` | Historical plans, experiment notes and code/evidence relocation manifests |

Reusable feature preparation and bootstrap code were extracted into `src/attribution_features.py` and `src/paired_analysis.py`; frozen ranking is in `scripts/evaluate_ranking.py`. Core plotting/evaluation no longer imports exploratory experiment entry points.

Historical locked artifacts retain their path text. Resolve old paths using [initial result relocation](../../results/archive/relocation_manifest.json), then [presentation relocation](../archive/presentation_relocation.json). [Code relocation](../archive/code_relocation.json) records script moves. File hashes were checked during evidence moves; no original data, test manifests or checkpoint weights changed. Historical documents are records, not current run instructions.

## Discontinued map-similarity extension

[Archive summary](../../results/archive/map_similarity/README.md): sparse/PCA codes, Pearson/cosine, dot products, projections and LGD/Pearson ablations, including the additional200-candidate cohort. Source: `scripts/archive/map_similarity/`; tests: `tests/archive/map_similarity/`; batch launcher: `hpc/archive/map_similarity/`. [Relocation manifest](../archive/map_similarity_relocation.json) records historical locations. The active study retains original L/G/D and its existing trends.
