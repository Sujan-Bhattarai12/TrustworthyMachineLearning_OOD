# Additional official-corruption generalization smoke test

Frozen expanded-study **ResNet-50** warning models, thresholds and checkpoint; no fitting, recalibration, rule search or outcome-based image selection. Four preselected additional official CIFAR-10-C types: **shot_noise, motion_blur, fog, jpeg_compression**. They represent noise, blur, weather-like and digital corruption. Existing Gaussian-blur observations serve as a reference.

A fixed label-stratified random subset of **200 expanded test candidates (20/class, seed20261117)** is reused across all types. Clean-correct selection and common attribution-validity exclusions reduce retained counts. These IDs were already part of the expanded blur evaluation; this is an exploratory frozen-method corruption extension, not fresh-image confirmation. Different corruption types still preserve the same10 class labels, so this evaluates corruption shift rather than open-set semantic OOD detection.

New maps use original clean predicted-class targets, IG64, raw Grad-CAM and original L/G/D formulas/trends, official severities0–5. Report all original primary/static/interaction methods and MSP/ODIN/confidence-trend baselines, plus existing frozen individual-L/G/D and entropy/Energy supplements. No Pearson, cosine, embedding or modified threshold rule is included. Output-only baseline forwards preserve the original loader tensor layout and require prediction/MSP consistency.

Every method uses the same valid trajectories within each corruption. Comparison.csv contains both per-corruption-valid and cross-corruption-common-image cohorts. Report counts and excluded.csv; do not claim paired image comparability across different retained subsets without using the common-image rows. All chosen types must be reported, including unfavorable outcomes. Small numbers of failing/nonfailing trajectories may make some rates uncertain or undefined.

Protocol.json locks model/specification hashes, subset manifest and official file hashes before map collection. Collection is resumable and saves all maps. Main expanded-study files are not altered.

Rivanna launcher: hpc/additional_corruptions.sbatch (oneA100,4h limit). Expected processing is roughly15–30minutes plus queue time based on prior ResNet throughput, but runtime is not guaranteed. Outputs: collection/, comparison.csv, excluded.csv, cohort.json, status.json. Reproduce/resume with `python -m scripts.run_corruption_panel_smoke run`.
