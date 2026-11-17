# Split manifests

## Final completed study

- `final_development_manifest.csv`: link to the authoritative 5,000 development-candidate manifest.
- `final_test_manifest.csv`: link to the authoritative 2,000 test-candidate manifest.

Both targets live in `results/expanded_cifar10c_5000dev_2000test/`. They contain candidate IDs, not images; valid cohort counts differ by classifier. Final runners use the authoritative result-directory paths directly. Keep targets with these links when copying the project.

## Historical files

`archive/historical_splits/` preserves the earlier 45,000 training IDs, 5,000 training-source development IDs, and 1,500-row v2 manifest. These are not final-study inputs. The final base classifiers were not trained on the 45,000-image partition.

`split_manifest_v2.csv` is only a compatibility link for older imports and default arguments. Final runs pass an explicit final manifest. Do not run the historical `scripts/create_split.py` to regenerate final manifests; it writes the legacy split paths.
