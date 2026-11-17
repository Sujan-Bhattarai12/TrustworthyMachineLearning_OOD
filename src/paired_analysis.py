"""Image-group paired bootstrap; conditional on existing fitted scores."""
import numpy as np
KEYS=["source","image_id","corruption","fold"]

def grouped_effects(candidate,reference,repetitions=2000,seed=42):
    if candidate.duplicated(KEYS).any() or reference.duplicated(KEYS).any():
        raise ValueError('Duplicate trajectory outcomes.')
    both=candidate.merge(reference,on=KEYS,suffixes=('_candidate','_control'),validate='one_to_one')
    if len(both)!=len(candidate) or len(both)!=len(reference):
        raise ValueError('Comparison cohorts must be identical.')
    a=both.first_failure_candidate.to_numpy()
    b=both.first_failure_control.to_numpy()
    if not np.allclose(a,b,equal_nan=True):
        raise ValueError('Failure labels disagree.')
    fail=np.isfinite(a)
    x=both[['source','image_id']].copy()
    x['failures']=fail.astype(float)
    x['negatives']=(~fail).astype(float)
    for suffix in ['candidate','control']:
        x[f'early_{suffix}']=both[f'early_warning_{suffix}'].astype(float).to_numpy()*fail
        x[f'lead_{suffix}']=both[f'credited_lead_steps_{suffix}'].to_numpy()*fail
        x[f'false_{suffix}']=both[f'false_alarm_{suffix}'].astype(float).to_numpy()*(~fail)
    grouped=x.groupby(['source','image_id']).sum()
    columns=list(grouped.columns)
    arr=grouped.to_numpy()
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(arr),size=(repetitions,len(arr)))
    sums=arr[indices].sum(axis=1)
    total=arr.sum(axis=0)
    def ratio(v,numerator,denominator):
        num=v[...,columns.index(numerator)]
        den=v[...,columns.index(denominator)]
        return np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=den>0)
    result=dict(n_image_groups=len(arr),n_trajectories=len(both),n_failures=int(fail.sum()),n_nonfailures=int((~fail).sum()))
    for name,num,den in [('early_warning_rate','early','failures'),('mean_lead','lead','failures'),('false_alarm_rate','false','negatives')]:
        candidate_value=float(ratio(total,f'{num}_candidate',den))
        control_value=float(ratio(total,f'{num}_control',den))
        diffs=ratio(sums,f'{num}_candidate',den)-ratio(sums,f'{num}_control',den)
        finite=diffs[np.isfinite(diffs)]
        lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
        result.update({f'{name}_candidate':candidate_value,f'{name}_control':control_value,
                       f'{name}_difference':candidate_value-control_value,
                       f'{name}_ci_low':float(lo),f'{name}_ci_high':float(hi)})
    return result

