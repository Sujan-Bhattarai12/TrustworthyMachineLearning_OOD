# Presentation guide: completed official study

Use [results/reporting/final/](../results/reporting/final/README.md). It contains exactly **four principal figures and three principal tables**, exported for slides and manuscript use. Historical custom-corruption example figures have been removed.

Suggested order:

1. Explain the question: can attribution change warn before first failure, and can development-calibrated rules generalize to an excluded corruption?
2. Show Table 1: three frozen architectures, actual fitting/calibration/test counts; distinguish candidate counts from valid images.
3. Explain the pipeline: official images → IG/Grad-CAM → L/G/D+trends → logistic classifier → frozen threshold. Map generators are not baselines; logistic regression is the secondary classifier.
4. Show Figure 1 and Table 2: warning sensitivity together with false alarms. Do not rank methods using early warnings alone.
5. Show Figure 2: next-step AUROC/AP explain score discrimination separately from threshold transfer. A strong ranking score can still have a conservative or poorly transferred cutoff.
6. Show Figure 3 and Table 3: individual L/G/D and combined/static/dynamic/interaction results. Combining metrics is not consistently best.
7. Show Figure 4: exploratory ResNet-only additional-corruption panel on187 shared images; shot-noise sensitivity gains have false-alarm costs and other baselines remain competitive.
8. Discuss limitations and the bounded conclusion: no consistent attribution advantage in this evaluated setting. This is a research finding, not proof that attribution cannot ever help.

Captions are in the final package README. Full method details and paper sources are in [FINAL_STUDY.md](FINAL_STUDY.md); metric interpretation in [METRICS_GUIDE.md](METRICS_GUIDE.md); script walkthrough in [CODE_GUIDE.md](CODE_GUIDE.md).

Week5 work remains qualitative analysis of successful warnings, misses and false alarms, and lead in severity steps. The four aggregate figures do not themselves constitute a case-study analysis. Week6 includes finishing narrative discussion, references, slides and reproducibility submission.

Historical attribution completeness/stability evidence remains under results/validation and figures/archive/historical_validation. Those earlier checks used different checkpoint settings, including WideResNet/RobustBench and historical50-step IG. Do not present them as newly performed validation for every final checkpoint or claim IG completeness applies unchanged after absolute-value aggregation; Grad-CAM has no equivalent completeness axiom.

Qualitative severity examples: use [figures/official_examples/](../figures/official_examples/README.md), which shows the same image across official Gaussian-blur levels for all three final classifiers. Manual-severity illustrations have been removed.
