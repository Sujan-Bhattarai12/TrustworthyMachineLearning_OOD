"""Evaluate copied frozen map-Pearson models on additional reserved images; no fitting."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from src.attribution_features import add_features
from scripts.evaluate_early_warnings import TRAJECTORY,trajectory_outcomes

ROOT=Path('results/fresh_map_pearson_200')

def similarities(data,method):
    pearson=np.full(len(data),np.nan);cosine=np.full(len(data),np.nan)
    for row in data.itertuples():
        with np.load(ROOT/'collection/maps'/f'{row.source}_{row.image_id}.npz',allow_pickle=False) as maps:
            a=maps[f'{row.corruption}_0_{method}'].astype(np.float64).reshape(-1)
            b=maps[f'{row.corruption}_{row.severity}_{method}'].astype(np.float64).reshape(-1)
        a=a/max(float(np.max(np.abs(a))),np.finfo(float).tiny)
        b=b/max(float(np.max(np.abs(b))),np.finfo(float).tiny)
        denom=np.linalg.norm(a)*np.linalg.norm(b)
        if denom>1e-12:cosine[row.Index]=np.clip(a@b/denom,-1,1)
        a=a-a.mean();b=b-b.mean();denom=np.linalg.norm(a)*np.linalg.norm(b)
        if denom>1e-12:pearson[row.Index]=np.clip(a@b/denom,-1,1)
    return pearson,cosine


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for file,digest in protocol['sha256'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=digest:raise ValueError(f'Frozen file changed: {file}')
    if not (ROOT/'collection/summary.csv').exists():raise ValueError('Collection incomplete.')
    raw=pd.read_csv(ROOT/'collection/images.csv')
    if set(raw.dataset)!={'cifar10c'} or set(raw.ig_steps)!={64} or set(raw.corruption)!={'gaussian_blur'}:raise ValueError('Unexpected collection settings.')
    data,initial_excluded=add_features(raw);data=data.reset_index(drop=True)
    similarities_by_method={};bad=np.zeros(len(data),dtype=bool)
    for method in ['ig','gradcam']:
        p,c=similarities(data,method)
        similarities_by_method[(method,'Pearson clean similarity')]=p
        similarities_by_method[(method,'Cosine clean similarity')]=c
        bad|=~np.isfinite(p)|~np.isfinite(c)
    excluded=data.loc[bad,TRAJECTORY].drop_duplicates()
    keep=~pd.MultiIndex.from_frame(data[TRAJECTORY]).isin(pd.MultiIndex.from_frame(excluded))
    selected=data.loc[keep].copy();frozen=json.loads((ROOT/'frozen/frozen_specifications.json').read_text())
    result=[];outcomes=[]
    for spec in frozen['methods']:
        method=spec['method'];name=spec['representation']
        if method=='baseline':scores=selected[spec['column']].to_numpy()
        else:
            model=joblib.load(ROOT/'frozen'/f'{method}_{name.split()[0].lower()}_warning.joblib')
            scores=model.predict_proba(similarities_by_method[(method,name)][keep].reshape(-1,1))[:,1]
        trajectory=trajectory_outcomes(selected,scores,spec['threshold'])
        fail=trajectory[trajectory.first_failure.notna()];nonfail=trajectory[trajectory.first_failure.isna()]
        mask=selected.eligible_fit.to_numpy();y=selected.loc[mask,'fails_next_step'];s=scores[mask]
        result.append(dict(method=method,representation=name,n_trajectories=len(trajectory),n_failures=len(fail),n_nonfailures=len(nonfail),
            n_early=int(fail.early_warning.sum()),n_false=int(nonfail.false_alarm.sum()),early_warning_rate=fail.early_warning.mean(),
            false_alarm_rate=nonfail.false_alarm.mean(),mean_lead_all_failures=fail.credited_lead_steps.mean(),
            auroc=roc_auc_score(y,s) if y.nunique()==2 else np.nan,aupr=average_precision_score(y,s) if y.nunique()==2 else np.nan))
        outcomes.append(trajectory.assign(method=method,representation=name))
    pd.DataFrame(result).to_csv(ROOT/'comparison.csv',index=False);pd.concat(outcomes).to_csv(ROOT/'trajectories.csv',index=False)
    excluded.to_csv(ROOT/'undefined_correlation_exclusions.csv',index=False);initial_excluded.to_csv(ROOT/'initial_exclusions.csv',index=False)
    (ROOT/'cohort.json').write_text(json.dumps(dict(candidates=200,clean_correct=int(raw.image_id.nunique()),base_valid_trajectories=int(data.image_group.nunique()),
        undefined_correlation_exclusions=len(excluded),retained_trajectories=int(selected.image_group.nunique())),indent=2))
    print(pd.DataFrame(result).to_string(index=False),flush=True)

if __name__=='__main__':main()
