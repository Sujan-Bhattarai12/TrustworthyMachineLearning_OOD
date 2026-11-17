# Local code and HPC synchronization

## Current status — November 16, 2026

The completed Rivanna code and results were reconciled locally on November 15. Subsequent cleanup is **local only** and has not been synchronized back to Rivanna. Do not assume the two current directory layouts match.

The current local `scripts/` contains only the eight entries listed in [scripts/README.md](../scripts/README.md). Historical entry points and validation scripts are archived; obsolete root links were removed. The supplementary evaluator imports helpers from the expanded runner, and the additional-corruption entry point is named `run_corruption_generalization.py`.

`configs/final_code_sha256.json` describes the cleaned current sources. Original source snapshots, reconciliation logs and relocation records are under [archive/](archive/README.md). The full test suite passed 89 tests after script cleanup.

## What cleanup did not change

The final Hugging Face checkpoints, official data, candidate manifests, collected maps, fitted warning models, thresholds and completed scientific results remain unchanged. Frozen protocol JSON files still describe their original execution sources. Guarded historical reruns require restoring those paths/bytes; do not alter historical hashes to bypass checks. Rebuilding the final figures from completed evaluation CSVs does not require a historical rerun.

The obsolete RobustBench WideResNet checkpoint was deleted at explicit user request. Its earlier validation checks require downloading it again.

## Future synchronization

Review the intended source/archival layout before syncing. Transfer retained relative links together with their targets. Keep local `.venv` and Linux `.venv-hpc` separate. Do not overwrite completed results or fitted artifacts during source cleanup.

The top-level `hpc/` directory was subsequently relocated locally to `docs/archive/hpc_launchers/`. No launchers were executed and no remote files changed. Historical launcher instructions assume the original layout.
