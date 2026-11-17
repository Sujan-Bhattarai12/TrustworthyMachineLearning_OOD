# Attribution validation

Preserved checks: custom IG versus Captum, IG completeness residual/convergence, Grad-CAM agreement and stability. These scripts were developed for the earlier RobustBench WideResNet setup and may require its optional dependencies/checkpoint. They are not the main final-model collection pipeline. Do not treat their historical results as fresh validation of every ResNet/ViT/Swin checkpoint.

Current IG rerun counts use powers of two (comparison default 64, convergence 16/32/64/128). Historical CSVs and images remain unchanged and may include a 50-step condition. No new validation was run during cleanup.

Historical evidence: `results/validation/`, `figures/validation/`. New stability reruns write separately named files rather than overwriting that evidence. Run from the project root with `python -m scripts.validation.<script_name>` only when a rerun is intended. Reading the existing evidence needs no model download.
