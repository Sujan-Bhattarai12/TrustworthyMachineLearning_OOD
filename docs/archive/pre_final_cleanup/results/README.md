# Results guide

## Presentation evidence

- `classifier_benchmarks_500/`: original 500-image development collections/comparisons for ResNet, ViT and Swin. Inspect method inputs before citing older comparison rows.
- `frozen_warning_methods/`: exact original selected frozen specifications.
- `final_blur_evaluation/`: original blur evaluation of those specifications, including candidate audits and excluded trajectories.
- `reporting/week4/`: consolidated cohort/threshold/timing/ranking CSVs and PNG/PDF figures.
- `reporting/attribution_error_relationships/`: latest IG/Grad-CAM error-association figures, counts and correlation/AUC tables.
- `validation/`: historical IG/Grad-CAM stability data and associated clean-image list. These predate the final checkpoint comparison.

## Archive

`archive/exploratory/` contains feature diagnosis, fusion, interactions, transfer and normalization/smoothing pilots. Other archive folders retain discontinued DRO/V-REx, hybrids and early smoke/pilot runs. `archive/logs/` contains main collection logs. Archived observations are not pooled with final findings.

No observations were deleted or modified during moves. Relocation records:

- `archive/relocation_manifest.json`: first result organization.
- `../docs/archive/audit_records/presentation_relocation.json`: subsequent presentation cleanup and hashes.

Resolve old paths in locked artifacts by applying these relocations in order. Test-image reservations still scan archived CSVs recursively. Original data and split manifests are unchanged.

See [presentation guide](../docs/PRESENTATION_GUIDE.md) and [Week 4 evaluation](../docs/WEEK4_EVALUATION.md).
