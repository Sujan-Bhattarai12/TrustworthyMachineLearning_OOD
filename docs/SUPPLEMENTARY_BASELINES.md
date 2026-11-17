# Final output-based supplementary baselines

The completed **expanded5000-development/2000-test-candidate study** includes deterministic predictive entropy and post-hoc Energy, alongside main MSP and ODIN. Actual retained image counts are architecture dependent. Entropy/Energy use the exact main-study common-valid cohorts and frozen development calibration groups; thresholds are saved before test scoring. They were added after the main protocol began and are identified as supplements.

- **Entropy:** H=−Σp_k ln(p_k), p=softmax(logits), temperature 1. Higher means greater predictive uncertainty. This is Shannon entropy applied to one deterministic predictive distribution, not MC Dropout. [Shannon,1948](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x) provides the information measure, not this project's detector.
- **Energy:** E=−T logΣexp(logits_k/T), T=1. Higher means greater warning score. Follow the post-hoc formula of [Liu, Wang, Owens and Li,2020](https://arxiv.org/abs/2010.03759), without their optional energy-based training.

Calibration uses maxima over nonfailing development calibration trajectories, target 5%, strict score>threshold. No test temperature search or threshold tuning. The scored quantities are adapted to our first-failure warning evaluation; their original OOD roles do not guarantee early-warning performance.

Code: src/supplementary_scores.py and scripts/evaluate_expanded_supplementary_baselines.py. Completed outputs: results/expanded_cifar10c_5000dev_2000test/supplementary_entropy_energy/. Inference preserves the collector's single-image loader tensor layout and enforces label/MSP consistency. Earlier batch/repacked-tensor implementations failed that guard before accepted results; the completed implementation passed it. [Execution history](archive/historical_notes/SUPPLEMENTARY_BASELINES_EXECUTION_HISTORY.md) is retained, not part of the final results.
