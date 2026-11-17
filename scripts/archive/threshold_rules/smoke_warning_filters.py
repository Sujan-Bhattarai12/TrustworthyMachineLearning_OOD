"""Calibration-selected 40% sensitivity thresholds and causal alarm filters."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from src.attribution_features import add_features
from scripts.evaluate_early_warnings import TRAJECTORY
from scripts.run_expanded_cifar10c_study import score_model,sha,save

ROOT=Path('results/warning_filter_smoke')
INPUT=Path('results/clustering_threshold_smoke/input')
PRIMARY=Path('results/expanded_cifar10c_5000dev_2000test')
TARGET=.4


def effective_scores(data,scores,rule):
    result=np.asarray(scores,dtype=float).copy()
    if rule=='Two consecutive':
        result[:]=-np.inf
        for _,g in data.groupby(TRAJECTORY,sort=False):
            ids=g.sort_values('severity').index.to_numpy()
            for prior,current in zip(ids[:-1],ids[1:]):
                if data.loc[prior,'severity']>=1 and data.loc[current,'severity']==data.loc[prior,'severity']+1:
                    result[current]=min(scores[prior],scores[current])
    elif rule.startswith('Rising loss'):
        method=rule.split(':')[1]
        result[data[f'delta_{method}_loss'].to_numpy()<=0]=-np.inf
    elif rule!='Threshold only':raise ValueError('Unknown rule.')
    result[data.severity.to_numpy()==0]=-np.inf
    return result


def trajectory_summary(data,effective):
    rows=[]
    for identity,g in data.groupby(TRAJECTORY,sort=False):
        ids=g.sort_values('severity').index.to_numpy();ff=g.first_failure.iloc[0]
        nonzero=ids[data.loc[ids,'severity'].to_numpy()>0]
        before=nonzero[data.loc[nonzero,'severity'].to_numpy()<ff] if np.isfinite(ff) else nonzero
        rows.append(dict(zip(TRAJECTORY,identity),first_failure=ff,maximum=float(np.max(effective[before])) if len(before) else -np.inf))
    return pd.DataFrame(rows)


def select_threshold(summary,target=TARGET):
    positive=summary.loc[summary.first_failure.notna(),'maximum'].to_numpy()
    finite=positive[np.isfinite(positive)]
    required=int(np.ceil(target*len(positive)))
    maximum=float(len(finite)/len(positive)) if len(positive) else 0.
    if required==0 or len(finite)<required:return dict(threshold=None,feasible=False,max_reachable_early_warning=maximum)
    boundary=np.sort(finite)[::-1][required-1]
    threshold=float(np.nextafter(boundary,-np.inf))
    negatives=summary.loc[summary.first_failure.isna(),'maximum'].to_numpy()
    return dict(threshold=threshold,feasible=True,max_reachable_early_warning=maximum,
        selection_early_warning=float(np.mean(positive>threshold)),selection_false_alarm=float(np.mean(negatives>threshold)))


def evaluate(data,effective,threshold):
    outcomes=[]
    for _,g in data.groupby(TRAJECTORY,sort=False):
        ff=g.first_failure.iloc[0]
        alarm=g[(g.severity>0)&(effective[g.index.to_numpy()]>threshold)].severity
        first=float(alarm.min()) if len(alarm) else np.nan
        early=bool(np.isfinite(ff) and np.isfinite(first) and first<ff)
        outcomes.append(dict(failure=np.isfinite(ff),early=early,false_alarm=bool(not np.isfinite(ff) and np.isfinite(first)),
            lead=float(ff-first) if early else 0.))
    frame=pd.DataFrame(outcomes);fail=frame[frame.failure];negative=frame[~frame.failure]
    return dict(n_failures=len(fail),n_nonfailures=len(negative),n_early=int(fail.early.sum()),n_false=int(negative.false_alarm.sum()),
        early_warning_rate=float(fail.early.mean()),false_alarm_rate=float(negative.false_alarm.mean()),mean_lead_all_failures=float(fail.lead.mean()))


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    folder=PRIMARY/'frozen/resnet50';artifact=json.loads((folder/'specifications.json').read_text())
    assert sha(INPUT/'development.csv')==artifact['development_sha256']
    dev,_=add_features(pd.read_csv(INPUT/'development.csv'));dev=dev.reset_index(drop=True)
    selection_ids,validation_ids=train_test_split(sorted(artifact['calibration_image_groups']),test_size=.5,random_state=43)
    selection=dev[dev.image_group.isin(selection_ids)].copy();validation=dev[dev.image_group.isin(validation_ids)].copy()
    # Reset independently for causal index-based rule evaluation.
    selection=selection.reset_index(drop=True);validation=validation.reset_index(drop=True)
    models=joblib.load(folder/'warning_models.joblib');specs=[];rows=[]
    for original in artifact['methods']:
        if original['name'] not in ['IG dynamic','GRADCAM dynamic','MSP']:continue
        scores=score_model(original,models,selection)
        rules=['Threshold only','Two consecutive']
        if original['name']!='MSP':rules.append('Rising loss:'+('ig' if original['name']=='IG dynamic' else 'gradcam'))
        for rule in rules:
            chosen=select_threshold(trajectory_summary(selection,effective_scores(selection,scores,rule)))
            specs.append(dict(model=original['name'],rule=rule,**chosen))
    save(ROOT/'frozen_thresholds.json',dict(scope='Exploratory reused blur cohorts; not untouched confirmation',target_early_warning=TARGET,
        selection_image_groups=selection_ids,validation_image_groups=validation_ids,
        model_sha256=sha(folder/'warning_models.joblib'),code_sha256=sha(__file__),methods=specs,
        rule_details='Strict score exceedance; persistence requires severities>=1 and credits second observation; rising loss requires deltaL>0',
        objective='Highest threshold with at least40% selection-trajectory early warning; minimizes false alarms among scalar thresholds; report infeasible target'))
    test,_=add_features(pd.read_csv(INPUT/'test.csv'));test=test.reset_index(drop=True)
    for cohort,data in [('development validation',validation),('Gaussian blur',test)]:
        for spec in specs:
            original=next(s for s in artifact['methods'] if s['name']==spec['model'])
            scores=score_model(original,models,data)
            if not spec['feasible']:
                rows.append(dict(cohort=cohort,model=spec['model'],rule=spec['rule'],feasible=False));continue
            effective=effective_scores(data,scores,spec['rule'])
            rows.append(dict(cohort=cohort,model=spec['model'],rule=spec['rule'],feasible=True,threshold=spec['threshold'],**evaluate(data,effective,spec['threshold'])))
        for original in artifact['methods']:
            if original['name'] not in ['IG dynamic','GRADCAM dynamic','MSP']:continue
            scores=score_model(original,models,data)
            rows.append(dict(cohort=cohort,model=original['name'],rule='Original 5%-target reference',feasible=True,threshold=original['threshold'],
                **evaluate(data,effective_scores(data,scores,'Threshold only'),original['threshold'])))
    result=pd.DataFrame(rows);result.to_csv(ROOT/'comparison.csv',index=False)
    print(result.to_string(index=False))

if __name__=='__main__':main()
