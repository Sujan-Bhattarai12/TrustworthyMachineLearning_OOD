"""Compare raw baselines, learned warnings, and individual ASWS feature ablations.

Run:
    python -m scripts.evaluate_warning_models

Evaluates:
    - Raw MSP and ODIN baselines
    - Logistic regression confidence comparator
    - Individual L, G, D models for IG and Grad-CAM
    - Combined L+G+D models for IG and Grad-CAM

Uses image-grouped cross-validation and shared prediction rows.

Primary output measures first-failure warning timing using separate fitting,
calibration, and evaluation images. Next-step AUROC/AUPR are secondary outputs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.warning_features import validate_attribution_columns
from scripts.archive.legacy_entrypoints.run_msp_baseline import build_warning_rows


METRICS = ["loss", "gain", "weighted_wasserstein"]

FEATURE_LABELS = {
    "loss": "L",
    "gain": "G",
    "weighted_wasserstein": "D",
}

METHODS = ("ig", "gradcam")


def evaluate(df):

    missing = {'odin'} - set(df.columns)
    if missing:
        raise ValueError(f'Missing benchmark scores {sorted(missing)}. Regenerate data with the revised classifier runner; historical CSVs cannot supply these scores.')

    # ---------------------------------------------------------
    # 1. Construct future-failure warning rows
    # ---------------------------------------------------------

    if 'classifier' in df and df.classifier.nunique() != 1:
        raise ValueError('Evaluate each classifier separately.')
    warnings = build_warning_rows(df)

    if warnings.empty:
        raise ValueError("No eligible next-severity examples.")

    keys = ["source", "image_id", "corruption", "severity"]

    # ---------------------------------------------------------
    # 2. Merge IG and Grad-CAM attribution features
    # ---------------------------------------------------------

    for method in METHODS:

        part = df.loc[
            df.method == method,
            keys + METRICS + ["valid_map"],
        ].copy()

        if part.duplicated(keys).any():
            raise ValueError(f"Duplicate {method} rows.")

        valid = (
            part.valid_map.astype(str)
            .str.lower()
            .isin(["true", "1"])
        )

        part.loc[~valid, METRICS] = np.nan

        part = part.drop(columns="valid_map")

        part = part.rename(
            columns={
                metric: f"{method}_{metric}"
                for metric in METRICS
            }
        )

        warnings = warnings.merge(
            part,
            on=keys,
            how="left",
            validate="one_to_one",
        )

    # ---------------------------------------------------------
    # 3. Add ODIN baseline
    # ---------------------------------------------------------

    predictions = df.drop_duplicates(keys)[
        keys + ["odin"]
    ]

    warnings = warnings.merge(
        predictions,
        on=keys,
        validate="one_to_one",
    )

    # ---------------------------------------------------------
    # 4. Keep common finite examples
    # ---------------------------------------------------------

    attribution_features = [
        f"{method}_{metric}"
        for method in METHODS
        for metric in METRICS
    ]

    required_features = (
        attribution_features
        + ["odin", "warning_score", "severity"]
    )

    finite = np.isfinite(
        warnings[required_features].to_numpy(dtype=float)
    ).all(axis=1)

    excluded = int((~finite).sum())

    warnings = warnings.loc[finite].reset_index(drop=True)

    if warnings.empty:
        raise ValueError(
            "No common finite attribution/baseline examples."
        )

    # ---------------------------------------------------------
    # 5. Group by image identity
    # ---------------------------------------------------------

    warnings["image_group"] = (
        warnings.source.astype(str)
        + ":"
        + warnings.image_id.astype(str)
    )

    y = warnings.fails_next_step.to_numpy(dtype=int)

    n_groups = warnings.image_group.nunique()

    if n_groups < 2:
        raise ValueError(
            "Need at least two image groups for cross-validation."
        )

    folds = list(
        GroupKFold(
            n_splits=min(5, n_groups)
        ).split(
            warnings,
            y,
            warnings.image_group,
        )
    )

    warnings["fold"] = -1

    for fold, (_, test) in enumerate(folds):
        warnings.loc[test, "fold"] = fold

    # ---------------------------------------------------------
    # 6. Raw baseline scores
    # ---------------------------------------------------------

    scores = {
        "MSP (raw)": warnings.warning_score.to_numpy(),
        "ODIN (raw)": warnings.odin.to_numpy(),
    }

    # ---------------------------------------------------------
    # 7. Define logistic regression models
    # ---------------------------------------------------------

    models = {
        "Confidence (logistic)": ["warning_score"],
    }

    for method in METHODS:

        prefix = f"Attribution L/G/D from {method.upper()}"

        # Individual feature ablations:
        # L only, G only, D only
        for metric in METRICS:

            label = FEATURE_LABELS[metric]

            models[f"{prefix}-{label}"] = [
                f"{method}_{metric}"
            ]

        # All three attribution features
        cols = [
            f"{method}_{metric}"
            for metric in METRICS
        ]

        models[f"{prefix} (L,G,D)"] = cols

    # ---------------------------------------------------------
    # 8. Train models using grouped cross-validation
    # ---------------------------------------------------------

    skipped = []

    for name, cols in models.items():
        if name.startswith("Attribution"):
            validate_attribution_columns(cols)

        # Every training fold needs both classes.
        if any(
            np.unique(y[train]).size < 2
            for train, _ in folds
        ):
            skipped.append(name)
            continue

        score = np.full(len(y), np.nan)

        for train, test in folds:

            pipeline = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42,
                ),
            )

            pipeline.fit(
                warnings.iloc[train][cols],
                y[train],
            )

            score[test] = pipeline.predict_proba(
                warnings.iloc[test][cols]
            )[:, 1]

        scores[name] = score

    # ---------------------------------------------------------
    # 9. Evaluate AUROC and AUPR
    # ---------------------------------------------------------

    summaries = []
    output = []

    corruptions = (
        ["all"]
        + sorted(warnings.corruption.unique().tolist())
    )

    for name, score in scores.items():

        result = warnings[
            keys + [
                "next_severity",
                "fails_next_step",
                "fold",
            ]
        ].copy()

        result["model"] = name
        result["score"] = score

        output.append(result)

        for corruption in corruptions:

            if corruption == "all":
                mask = np.ones(len(y), dtype=bool)
            else:
                mask = (
                    warnings.corruption == corruption
                ).to_numpy()

            labels = y[mask]
            values = score[mask]

            both_classes = (
                np.unique(labels).size == 2
            )

            summaries.append(
                {
                    "model": name,
                    "corruption": corruption,
                    "n_steps": len(labels),
                    "n_images": warnings.loc[
                        mask, "image_group"
                    ].nunique(),
                    "n_failures": int(labels.sum()),
                    "failure_rate": float(labels.mean()),
                    "auroc": (
                        roc_auc_score(labels, values)
                        if both_classes
                        else np.nan
                    ),
                    "aupr": (
                        average_precision_score(labels, values)
                        if both_classes
                        else np.nan
                    ),
                }
            )

    summary_df = pd.DataFrame(summaries)
    if 'classifier' in df:
        summary_df['classifier'] = df.classifier.iloc[0]
        for part in output:
            part['classifier'] = df.classifier.iloc[0]


    predictions_df = pd.concat(
        output,
        ignore_index=True,
    )

    return (
        summary_df,
        predictions_df,
        excluded,
        skipped,
    )


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/classifier_benchmarks_500/resnet50/images.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/development_comparison_attribution_only"
        ),
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    from scripts.evaluate_early_warnings import run as evaluate_timing
    print('PRIMARY EVALUATION: warning timing at a 5% calibration trajectory false-alarm target')
    primary = evaluate_timing(args.input, args.output_dir / 'early_warning')

    (
        summary,
        predictions,
        excluded,
        skipped,
    ) = evaluate(df)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary.to_csv(args.output_dir / 'summary.csv', index=False)
    summary.to_csv(
        args.output_dir / "secondary_next_step_summary.csv",
        index=False,
    )

    predictions.to_csv(
        args.output_dir / "secondary_next_step_predictions.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("SECONDARY: NEXT-STEP FAILURE RANKING")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nExcluded {excluded} steps with "
        "invalid/missing scores from every method."
    )

    if skipped:
        print(
            "\nSkipped learned models because "
            "a training fold contained only one class or a baseline is unavailable:"
        )
        print(", ".join(skipped))

    print("\nSaved results to:")
    print(args.output_dir / "summary.csv")
    print(args.output_dir / "early_warning/trajectories.csv")
    print(args.output_dir / "secondary_next_step_summary.csv")

    print(
        "\nTiming is the primary development result; AUROC/AUPR are secondary. "
        "Neither is a reserved final-test result. Compare achieved evaluation "
        "false-alarm rates before claiming one method warns earlier fairly."
    )


if __name__ == "__main__":
    main()
