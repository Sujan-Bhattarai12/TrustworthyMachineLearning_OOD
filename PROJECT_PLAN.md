# Final project status

The experimental work is complete. No new models, thresholds or exploratory methods are scheduled.

- [x] Official CIFAR-10-C expanded study: three architectures,5000 development/2000 test candidates, IG64.
- [x] Original attribution-only L/G/D + trends; static/interactions and individual L/G/D ablations.
- [x] MSP, ODIN, entropy and Energy; separate confidence-trend control.
- [x] Frozen-model additional-corruption panel on ResNet-50: shot noise, motion blur, fog, JPEG compression.
- [x] Completed results and maps synced locally.
- [x] Discarded encodings/similarities/projections and threshold experiments archived.
- [x] Final four figures and three tables prepared with captions and source hashes.
- [x] Code guide, study specification and metric definitions prepared.
- [ ] Finish narrative report and presentation using the final results package.

**Conclusion:** attribution-derived warning scores did not show a advantage over output-based baselines within the evaluated protocol. No single attribution component or combination is consistently best. Do not report only the best test-selected variant.

Use the [final report package](results/reporting/final/README.md), [study definition](docs/FINAL_STUDY.md), [metric guide](docs/METRICS_GUIDE.md), [code guide](docs/CODE_GUIDE.md), and [archive index](docs/ARCHIVE_INDEX.md). The earlier chronological plan is preserved in docs/archive/pre_final_cleanup/PROJECT_PLAN.md.

Main GPU job22982030 completed in16h 36m 01s; entropy/Energy22982067 in41m 12s; individual ablations23004665 in3m 06s; corruption panel23006811 in10m 41s. Failed diagnostic/supplement attempts are historical and not presented as results.
