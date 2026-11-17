"""Primary development evaluation: first-failure warning timing with nested image splits.

Outer test images never fit score weights/scales or calibrate thresholds. Within
outer-training images, a separate calibration subset sets trajectory-level FPR.
This is development cross-validation, not the reserved final blur/test evaluation.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.warning_features import validate_attribution_columns,warning_method_role

KEYS = ['source', 'image_id', 'corruption', 'severity']
TRAJECTORY = KEYS[:-1]
COMPONENTS = ['loss', 'gain', 'weighted_wasserstein']


def prepare(df):
    if 'classifier' in df and df.classifier.nunique() != 1:
        raise ValueError('Evaluate one classifier per input.')
    expected = set(range(6))
    required = KEYS + ['true_label', 'clean_class', 'predicted_class', 'msp', 'odin', 'method', 'valid_map'] + COMPONENTS
    missing = set(required) - set(df)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    shared = KEYS + ['true_label', 'clean_class', 'predicted_class', 'msp', 'odin']
    if (df.groupby(KEYS)[shared[4:]].nunique(dropna=False) > 1).any().any():
        raise ValueError('Attribution rows disagree on classifier predictions/scores.')
    data = df[shared].drop_duplicates(KEYS)
    features = []
    for method in ['ig', 'gradcam']:
        part = df[df.method == method][KEYS + COMPONENTS + ['valid_map']].copy()
        if part.duplicated(KEYS).any():
            raise ValueError(f'Duplicate {method} rows.')
        valid = part.valid_map.astype(str).str.lower().isin(['true', '1'])
        part.loc[~valid, COMPONENTS] = np.nan
        rename = {c: f'{method}_{c}' for c in COMPONENTS}
        features += list(rename.values())
        data = data.merge(part.drop(columns='valid_map').rename(columns=rename), on=KEYS, how='left', validate='one_to_one')
    kept, excluded = [], []
    for identity, group in data.groupby(TRAJECTORY):
        group = group.sort_values('severity').copy()
        reason = None
        if set(group.severity) != expected or len(group) != 6:
            reason = 'incomplete severity trajectory (requires 0 through 5)'
        elif not np.isfinite(group[features + ['msp', 'odin']]).all().all():
            reason = 'invalid score/attribution (whole trajectory excluded for every method)'
        elif group.true_label.nunique() != 1 or group.clean_class.nunique() != 1:
            reason = 'inconsistent class labels'
        elif group.iloc[0].predicted_class != group.iloc[0].true_label or group.iloc[0].clean_class != group.iloc[0].true_label:
            reason = 'not clean-correct'
        if reason:
            excluded.append(dict(zip(TRAJECTORY, identity), reason=reason))
            continue
        failed = group[group.predicted_class != group.true_label]
        first_failure = float(failed.severity.min()) if len(failed) else np.nan
        group['first_failure'] = first_failure
        group['image_group'] = group.source.astype(str) + ':' + group.image_id.astype(str)
        group['eligible_fit'] = (group.severity > 0) & (group.severity < (first_failure if len(failed) else 5))
        # Training labels use only the next step before the FIRST failure.
        group['fails_next_step'] = ((group.severity + 1) == first_failure).astype(int)
        kept.append(group)
    if not kept:
        raise ValueError('No complete, valid, clean-correct trajectories.')
    return pd.concat(kept, ignore_index=True), pd.DataFrame(excluded, columns=TRAJECTORY + ['reason'])


def calibrate_threshold(maxima, target_fpr):
    """Smallest observed threshold with <= target empirical FPR; alarm uses >.

    Ties are conservative. With very few negatives, only zero empirical false
    alarms may be attainable. No negative trajectories means no calibration.
    """
    values = np.asarray(maxima, dtype=float)
    if not len(values):
        return np.nan
    for threshold in np.unique(np.sort(values)):
        if np.mean(values > threshold) <= target_fpr:
            return float(threshold)
    raise AssertionError('Maximum threshold must give zero empirical FPR.')


def trajectory_outcomes(data, scores, threshold):
    scored = data.copy()
    scored['score'] = scores
    rows = []
    for identity, group in scored.groupby(TRAJECTORY):
        failure = group.first_failure.iloc[0]
        observed = group[(group.severity > 0) & (group.score > threshold)]
        first_alarm = float(observed.severity.min()) if len(observed) else np.nan
        early = bool(np.isfinite(failure) and np.isfinite(first_alarm) and first_alarm < failure)
        lead = float(failure - first_alarm) if early else 0.0
        rows.append(dict(zip(TRAJECTORY, identity), first_failure=failure,
                         first_alarm=first_alarm, early_warning=early,
                         missed_warning=bool(np.isfinite(failure) and not early),
                         false_alarm=bool(not np.isfinite(failure) and np.isfinite(first_alarm)),
                         credited_lead_steps=lead,
                         detected_lead_steps=lead if early else np.nan))
    return pd.DataFrame(rows)


def evaluate(data, target_fpr=0.05, n_splits=5, seed=42, extra_scores=None):
    groups = data.image_group
    if groups.nunique() < 6:
        raise ValueError('Need at least six image groups for separate fit/calibration/test sets.')
    folds = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    outcomes, thresholds, memberships = [], [], []
    for fold, (development_idx, test_idx) in enumerate(folds.split(data, groups=groups)):
        development = data.iloc[development_idx]
        test = data.iloc[test_idx]
        fit_ids, calibration_ids = train_test_split(sorted(development.image_group.unique()), test_size=0.25, random_state=seed + fold)
        fit = development[development.image_group.isin(fit_ids)]
        calibration = development[development.image_group.isin(calibration_ids)]
        for role, subset in [('fit', fit), ('calibration', calibration), ('evaluation', test)]:
            for image in sorted(subset.image_group.unique()):
                memberships.append(dict(fold=fold, image_group=image, role=role))
        fit_rows = fit[fit.eligible_fit]
        score_sets = {'MSP': (1 - calibration.msp.to_numpy(), 1 - test.msp.to_numpy()),
                      'ODIN': (calibration.odin.to_numpy(), test.odin.to_numpy())}
        if fit_rows.empty:
            # Raw components remain evaluable; fixed/learned scores need fitting rows.
            fit_rows = fit[fit.severity > 0]
        for method in ['ig', 'gradcam']:
            columns = validate_attribution_columns([f'{method}_{c}' for c in COMPONENTS])
            for col, label in zip(columns, ['L', 'G', 'D']):
                score_sets[f'{method.upper()} {label} raw'] = (calibration[col].to_numpy(), test[col].to_numpy())
            # Scale by fit-only 95th percentile; no clipping or test-based rescaling.
            scale = fit_rows[columns].quantile(0.95).to_numpy()
            scale = np.where(scale > 1e-12, scale, 1.0)
            for size in [2, 3]:
                for indices in combinations(range(3), size):
                    chosen = [columns[i] for i in indices]
                    labels = '+'.join('LGD'[i] for i in indices)
                    cs = (calibration[chosen].to_numpy() / scale[list(indices)]).mean(axis=1)
                    ts = (test[chosen].to_numpy() / scale[list(indices)]).mean(axis=1)
                    score_sets[f'{method.upper()} {labels} fixed'] = (cs, ts)
            training = fit[fit.eligible_fit]
            if training.fails_next_step.nunique() == 2:
                pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
                pipe.fit(training[columns], training.fails_next_step)
                score_sets[f'{method.upper()} L+G+D learned'] = (pipe.predict_proba(calibration[columns])[:, 1], pipe.predict_proba(test[columns])[:, 1])
            else:
                thresholds.append(dict(fold=fold, model=f'{method.upper()} L+G+D learned', status='skipped: fit labels have one class'))
        if extra_scores is not None:
            additions = extra_scores(fit, calibration, test)
            if set(additions) & set(score_sets):
                raise ValueError('Extra score names must not replace original methods.')
            score_sets.update(additions)
        for model, (cal_scores, test_scores) in score_sets.items():
            if warning_method_role(model) == 'historical_hybrid':
                raise ValueError('Active attribution-only evaluation rejects confidence/attribution hybrid scores.')
            cal = calibration.copy()
            cal['score'] = cal_scores
            negatives = cal[cal.first_failure.isna() & (cal.severity > 0)]
            maxima = negatives.groupby(TRAJECTORY).score.max().to_numpy()
            threshold = calibrate_threshold(maxima, target_fpr)
            record = dict(fold=fold, model=model, threshold=threshold,
                          n_calibration_nonfailure_trajectories=len(maxima), target_fpr=target_fpr,
                          calibration_fpr=float(np.mean(maxima > threshold)) if len(maxima) else np.nan,
                          status='ok' if len(maxima) else 'skipped: no nonfailure calibration trajectories')
            thresholds.append(record)
            if not len(maxima):
                continue
            result = trajectory_outcomes(test, test_scores, threshold)
            result['model'], result['fold'], result['threshold'] = model, fold, threshold
            outcomes.append(result)
    if not outcomes:
        raise ValueError('No calibratable folds; more nonfailure calibration trajectories are needed.')
    trajectories = pd.concat(outcomes, ignore_index=True)
    trajectories['score_family'] = trajectories.model.map(warning_method_role)
    summaries = []
    for model, group in trajectories.groupby('model'):
        for corruption in ['all'] + sorted(group.corruption.unique()):
            g = group if corruption == 'all' else group[group.corruption == corruption]
            failures, negatives = g[g.first_failure.notna()], g[g.first_failure.isna()]
            summaries.append(dict(model=model, score_family=warning_method_role(model), corruption=corruption, n_trajectories=len(g),
                                  n_failures=len(failures), n_nonfailures=len(negatives),
                                  n_early_warnings=int(failures.early_warning.sum()),
                                  n_missed_warnings=int(failures.missed_warning.sum()),
                                  early_warning_rate=float(failures.early_warning.mean()) if len(failures) else np.nan,
                                  mean_lead_all_failures=float(failures.credited_lead_steps.mean()) if len(failures) else np.nan,
                                  mean_lead_detected_only=float(failures.detected_lead_steps.mean()),
                                  false_alarm_rate=float(negatives.false_alarm.mean()) if len(negatives) else np.nan,
                                  target_calibration_fpr=target_fpr))
    paired = []
    for baseline in ['MSP', 'ODIN']:
        reference = trajectories[trajectories.model == baseline]
        for model, group in trajectories.groupby('model'):
            if model in ['MSP', 'ODIN']:
                continue
            joined = group.merge(reference, on=TRAJECTORY + ['fold'], suffixes=('', '_baseline'), validate='one_to_one')
            joined = joined[joined.first_failure.notna()]
            delta = joined.credited_lead_steps - joined.credited_lead_steps_baseline
            paired.append(dict(model=model, baseline=baseline, n_shared_failures=len(joined),
                               n_earlier=int((delta > 0).sum()), n_tied=int((delta == 0).sum()),
                               n_later=int((delta < 0).sum()), mean_credited_lead_difference=float(delta.mean())))
    return pd.DataFrame(summaries), trajectories, pd.DataFrame(thresholds), pd.DataFrame(memberships), pd.DataFrame(paired)


def run(input_path, output_dir, target_fpr=0.05):
    df = pd.read_csv(input_path)
    data, excluded = prepare(df)
    summary, trajectories, thresholds, splits, paired = evaluate(data, target_fpr)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [('summary', summary), ('trajectories', trajectories), ('thresholds', thresholds),
                        ('splits', splits), ('paired_comparisons', paired), ('excluded_trajectories', excluded)]:
        if 'classifier' in df:
            frame = frame.assign(classifier=df.classifier.iloc[0])
        frame.to_csv(output_dir / f'{name}.csv', index=False)
    (output_dir / 'protocol.json').write_text(json.dumps(dict(primary='first-failure early warning',
        split='outer image-grouped evaluation; inner disjoint fit and calibration images',
        target_fpr=target_fpr, threshold_rule='score > threshold',
        fpr_unit='any alarm on a nonfailing trajectory across nonzero severities',
        lead_unit='severity steps; zero credited lead for missed/late warnings',
        scaling='fit-only component 95th percentile for equal-weight combinations',
        note='Development diagnostic; not reserved final test. Nominal calibration FPR does not guarantee equal evaluation FPR.'), indent=2))
    print(summary[summary.corruption == 'all'].to_string(index=False))
    print(f'Excluded {len(excluded)} whole trajectories. Saved primary evaluation to {output_dir}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--target-fpr', type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.target_fpr < 1:
        parser.error('Target FPR must be in [0,1).')
    run(args.input, args.output_dir, args.target_fpr)


if __name__ == '__main__':
    main()
