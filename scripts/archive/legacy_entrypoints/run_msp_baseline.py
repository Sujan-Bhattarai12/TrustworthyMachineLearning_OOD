"""Evaluate the Hendrycks–Gimpel confidence score for next-step failure.

Run from the project root: python -m scripts.run_msp_baseline
Reference: https://arxiv.org/abs/1610.02136

The published score detects current errors. Here it is adapted to predict
failure at the next sampled severity, using only currently correct distorted
images. No warning model is fitted and no test-set threshold is selected.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def build_warning_rows(df):
    required = {"source", "image_id", "severity", "clean_class", "true_label",
                "predicted_class", "target_confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    # Separate corruption trajectories when the input contains multiple types.
    groups = ["source", "image_id"]
    if "corruption" in df.columns:
        groups.append("corruption")
    keys = groups + ["severity"]
    prediction_cols = ["clean_class", "true_label", "predicted_class", "target_confidence"]
    if df[keys + prediction_cols].isna().any().any():
        raise ValueError("Image identities, labels, severities, and confidence must be present.")
    if not np.isfinite(df["target_confidence"]).all() or not df["target_confidence"].between(0, 1).all():
        raise ValueError("Confidence must be finite and in [0, 1].")
    # IG and Grad-CAM rows share predictions; count each image/severity once.
    if (df.groupby(keys)[prediction_cols].nunique() > 1).any().any():
        raise ValueError("Duplicate image/severity rows disagree on predictions or confidence.")
    unique = df.drop_duplicates(keys)
    rows = []
    for _, trajectory in unique.groupby(groups):
        trajectory = trajectory.sort_values("severity")
        records = trajectory.to_dict("records")
        if trajectory["true_label"].nunique() != 1 or trajectory["clean_class"].nunique() != 1:
            raise ValueError("Labels and clean class must stay fixed within a trajectory.")
        for current, following in zip(records, records[1:]):
            # On these eligible rows the clean target is the current argmax,
            # so stored target_confidence is exactly maximum softmax probability.
            if (current["severity"] <= 0
                    or current["clean_class"] != current["true_label"]
                    or current["predicted_class"] != current["true_label"]):
                continue
            row = {key: current[key] for key in groups}
            row.update(severity=current["severity"],
                       next_severity=following["severity"],
                       msp=current["target_confidence"],
                       warning_score=1.0 - current["target_confidence"],
                       fails_next_step=int(following["predicted_class"] != following["true_label"]))
            rows.append(row)
    return pd.DataFrame(rows, columns=groups + ["severity", "next_severity", "msp",
                                               "warning_score", "fails_next_step"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/classifier_benchmarks_500/resnet50/images.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/msp_baseline"))
    args = parser.parse_args()
    warnings = build_warning_rows(pd.read_csv(args.input))
    if warnings.empty:
        raise ValueError("No eligible currently correct distorted steps with a following severity.")
    y = warnings["fails_next_step"]
    both_classes = y.nunique() == 2
    summary = dict(n_images=len(warnings.drop_duplicates(["source", "image_id"])),
                   n_steps=len(warnings), n_failures=int(y.sum()),
                   failure_rate=float(y.mean()),
                   auroc=roc_auc_score(y, warnings["warning_score"]) if both_classes else np.nan,
                   aupr=average_precision_score(y, warnings["warning_score"]) if both_classes else np.nan)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warnings.to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))
    if not both_classes:
        print("AUROC/AUPR omitted: evaluation needs both failure and non-failure examples.")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
