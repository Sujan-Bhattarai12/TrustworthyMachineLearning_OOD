# Archive index

Archives preserve research history and reproducibility. They are not final-study methods or results.

| Location | Contents |
|---|---|
| `scripts/archive/legacy_entrypoints/` | Earlier runners, reports and warning-evaluation implementations |
| `scripts/archive/validation/` | Historical IG/Grad-CAM stability, agreement and completeness diagnostics |
| Other `scripts/archive/` folders | Discontinued feature, similarity and threshold experiments |
| `splits/archive/historical_splits/` | Earlier training/development/test manifests |
| `docs/archive/hpc_launchers/` | Rivanna launchers, environment records and earlier HPC scripts |
| [docs/archive/](archive/README.md) | Historical notes, audits and source backups |

The final study is specified in [FINAL_STUDY.md](FINAL_STUDY.md); use [the final results package](../results/reporting/final/README.md) for reporting.

The legacy custom-corruption generator is archived, with a relative compatibility link at `src/corruptions.py`. Historical split compatibility is explained in `splits/README.md`. Obsolete script root links have been removed. Frozen source guards require original paths/bytes for historical reruns.

The former checkpoint archive was explicitly deleted; no RobustBench checkpoint remains locally. Its hash record is provenance only.

Superseded `results/archive/` and archived tests were removed during cleanup. Prior possible test-image exposure IDs remain in `results/validation/prior_experiment_test_ids.csv`; precursor setup records remain in `results/final_cifar10c/`.
