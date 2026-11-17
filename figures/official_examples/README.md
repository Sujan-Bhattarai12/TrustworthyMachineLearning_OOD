# Official same-image attribution examples

Each model figure shows the **same original test image, ID 13**, at severity 0 (clean) and official CIFAR-10-C Gaussian-blur severities 1–5. Rows show image pixels, IG maps and Grad-CAM maps. All three model figures use the same image ID.

Images are loaded directly from official arrays; no manual corruption generator is used. Maps and predictions are reused from the completed final run, with IG64 and the fixed clean predicted target. Each map row uses its clean-map maximum as the scale reference and a shared color range across severities, preserving magnitude changes. Color scales can differ between methods/models; do not compare their raw magnitudes directly.

Selection is the lowest common saved test ID, independent of score or success. This example is illustrative, not proof of typical behavior. Source paths/hashes and selection are in `provenance.json`.

Rebuild: `MPLCONFIGDIR=/tmp/asws-final-mpl .venv/bin/python -m scripts.build_official_attribution_examples`. No inference or fitting is performed.

Aggregate performance remains in `results/reporting/final/`; these qualitative figures do not replace it.
