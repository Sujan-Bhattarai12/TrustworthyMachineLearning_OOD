"""Supplementary single-component attribution ablations on expanded study cohorts."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from src.attribution_features import add_features
from scripts.run_expanded_cifar10c_study import CLASSIFIERS,check,sha,save,make_model
from scripts.evaluate_early_warnings import TRAJECTORY,COMPONENTS,calibrate_threshold,trajectory_outcomes


def recipes():
    return [dict(name=f'{method.upper()} {label} alone',column=f'{method}_{component}',method=method,label=label)
            for method in ['ig','gradcam'] for label,component in zip('LGD',COMPONENTS)]


def fit_components(data,artifact):
    fit_ids=artifact['fit_image_groups'];cal_ids=artifact['calibration_image_groups']
    if set(fit_ids)&set(cal_ids):raise ValueError('Overlapping original partitions.')
    if set(data.image_group)!=set(fit_ids)|set(cal_ids):raise ValueError('Development cohort differs from original frozen groups.')
    training=data[data.image_group.isin(fit_ids)&data.eligible_fit];cal=data[data.image_group.isin(cal_ids)]
    models={};specs=[]
    for recipe in recipes():
        model=make_model(False).fit(training[[recipe['column']]],training.fails_next_step)
        scores=model.predict_proba(cal[[recipe['column']]])[:,1]
        c=cal.assign(score=scores)
        maxima=c[c.first_failure.isna()&(c.severity>0)].groupby(TRAJECTORY).score.max().to_numpy()
        if not len(maxima):raise ValueError('No nonfailing calibration trajectories.')
        threshold=float(calibrate_threshold(maxima,.05))
        models[recipe['name']]=model
        specs.append(dict(**recipe,threshold=threshold,calibration_negatives=len(maxima),calibration_fpr=float(np.mean(maxima>threshold)),
            standardized_weight=float(model[-1].coef_[0,0]),intercept=float(model[-1].intercept_[0])))
    return models,specs


def setup(root):
    out=root/'individual_lgd_ablations';path=out/'protocol.json'
    if path.exists():raise ValueError('Supplementary ablation protocol already exists.')
    save(path,dict(primary_protocol_sha256=sha(root/'locked_protocol.json'),added_unix=time.time(),
        scope='Supplementary ablations added during expanded run after initial ResNet results; report separately, no test-based winner selection',
        recipes=recipes(),inputs='Single raw L, G or retention-weighted Wasserstein D; no trends, interactions or confidence inputs',
        warning_model='StandardScaler + LogisticRegression C=1 max_iter2000 random_state42, matching primary model family',
        cohorts='Identical primary common-valid cohorts and original frozen fit/calibration image groups; no new split',
        calibration='Nonfailing calibration trajectory maxima, target5%, strict exceedance; freeze before reading test observations',
        code_sha256={str(Path(__file__)):sha(__file__)}))


def run(root):
    out=root/'individual_lgd_ablations';protocol=json.loads((out/'protocol.json').read_text())
    check(root)
    if sha(root/'locked_protocol.json')!=protocol['primary_protocol_sha256']:raise ValueError('Primary protocol changed.')
    for file,digest in protocol['code_sha256'].items():
        if sha(file)!=digest:raise ValueError('Supplementary code changed.')
    all_results=[]
    for classifier in CLASSIFIERS:
        original=root/'frozen'/classifier;lock=json.loads((original/'lock.json').read_text())
        if sha(original/'specifications.json')!=lock['specifications_sha256']:raise ValueError('Original specification changed.')
        artifact=json.loads((original/'specifications.json').read_text())
        source=root/'development'/classifier/'images.csv'
        if sha(source)!=artifact['development_sha256']:raise ValueError('Development source changed.')
        dev,_=add_features(pd.read_csv(source));folder=out/classifier;folder.mkdir(parents=True,exist_ok=True)
        frozen=folder/'frozen_specifications.json'
        if frozen.exists():
            specs_record=json.loads(frozen.read_text())
            if sha(folder/'models.joblib')!=specs_record['models_sha256'] or specs_record['primary_specifications_sha256']!=sha(original/'specifications.json'):
                raise ValueError('Frozen individual-component artifacts changed.')
            models=joblib.load(folder/'models.joblib');specs=specs_record['methods']
        else:
            models,specs=fit_components(dev,artifact);joblib.dump(models,folder/'models.joblib')
            save(frozen,dict(methods=specs,models_sha256=sha(folder/'models.joblib'),primary_specifications_sha256=sha(original/'specifications.json'),
                supplementary_protocol_sha256=sha(out/'protocol.json')))
        # Test observations are loaded only after this classifier's weights/thresholds are frozen.
        data,_=add_features(pd.read_csv(root/'test'/classifier/'images.csv'))
        original_trajectories=pd.read_csv(root/'evaluation'/classifier/'trajectories.csv')
        expected=set(map(tuple,original_trajectories[TRAJECTORY].drop_duplicates().to_numpy()))
        if set(map(tuple,data[TRAJECTORY].drop_duplicates().to_numpy()))!=expected:raise ValueError('Test cohort mismatch.')
        rows=[];trajectories=[]
        for spec in specs:
            scores=models[spec['name']].predict_proba(data[[spec['column']]])[:,1]
            trajectory=trajectory_outcomes(data,scores,spec['threshold'])
            fail=trajectory[trajectory.first_failure.notna()];nonfail=trajectory[trajectory.first_failure.isna()]
            eligible=data.eligible_fit.to_numpy();y=data.loc[eligible,'fails_next_step'];s=scores[eligible]
            rows.append(dict(classifier=classifier,model=spec['name'],reporting_scope='supplementary individual-component ablation',
                n_trajectories=len(trajectory),n_failures=len(fail),n_nonfailures=len(nonfail),n_early=int(fail.early_warning.sum()),
                early_warning_rate=fail.early_warning.mean(),missed_warning_rate=fail.missed_warning.mean(),
                false_alarm_rate=nonfail.false_alarm.mean(),mean_lead_all_failures=fail.credited_lead_steps.mean(),
                auroc=roc_auc_score(y,s) if y.nunique()==2 else np.nan,aupr=average_precision_score(y,s) if y.nunique()==2 else np.nan))
            trajectories.append(trajectory.assign(classifier=classifier,model=spec['name']))
        result=pd.DataFrame(rows);result.to_csv(folder/'comparison.csv',index=False);all_results.append(result)
        pd.concat(trajectories).to_csv(folder/'trajectories.csv',index=False)
        primary=pd.read_csv(root/'evaluation'/classifier/'comparison.csv').assign(reporting_scope='primary protocol')
        pd.concat([primary,result],ignore_index=True).to_csv(folder/'comparison_with_primary.csv',index=False)
        print(f'Completed individual L/G/D ablations: {classifier}',flush=True)
    pd.concat(all_results).to_csv(out/'comparison_all_classifiers.csv',index=False)
    save(out/'status.json',dict(status='complete',finished_unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['setup','run'])
    parser.add_argument('--root',type=Path,default=Path('results/expanded_cifar10c_5000dev_2000test'))
    args=parser.parse_args();(setup if args.action=='setup' else run)(args.root)
