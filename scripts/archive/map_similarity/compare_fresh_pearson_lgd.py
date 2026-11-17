"""Compare frozen pilot L/G/D+trends with Pearson on identical fresh trajectories."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from src.attribution_features import add_features
from scripts.evaluate_early_warnings import TRAJECTORY,COMPONENTS,trajectory_outcomes

ROOT=Path('results/fresh_map_pearson_200')
PILOT=Path('results/local_sparse_map_pilot')

def main():
    data,_=add_features(pd.read_csv(ROOT/'images.csv'))
    recorded=pd.read_csv(ROOT/'trajectories.csv')
    cohorts=[]
    for _,g in recorded.groupby(['method','representation']):
        cohorts.append(set(map(tuple,g[TRAJECTORY].to_numpy())))
    assert cohorts and all(c==cohorts[0] for c in cohorts)
    keep=pd.MultiIndex.from_frame(data[TRAJECTORY]).isin(pd.MultiIndex.from_tuples(sorted(cohorts[0]),names=TRAJECTORY))
    data=data.loc[keep].copy()
    assert data.image_group.nunique()==168
    thresholds=pd.read_csv(PILOT/'comparison.csv')
    locked=json.loads((PILOT/'gaussian_blur_transfer/frozen_transfer_protocol.json').read_text())
    results=[];outcomes=[];evidence={}
    for method in ['ig','gradcam']:
        path=PILOT/f'{method}_L_G_D_warning.joblib'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==locked['frozen_sha256'][str(path)]
        evidence[str(path)]=digest
        spec=thresholds[(thresholds.method==method)&(thresholds.representation=='L/G/D + trends')].iloc[0]
        columns=[f'{method}_{c}' for c in COMPONENTS]
        columns += [f'delta_{key}' for key in columns.copy()]
        model=joblib.load(path);scores=model.predict_proba(data[columns].to_numpy())[:,1]
        outcome=trajectory_outcomes(data,scores,spec.threshold)
        assert set(map(tuple,outcome[TRAJECTORY].to_numpy()))==cohorts[0]
        fail=outcome[outcome.first_failure.notna()];nonfail=outcome[outcome.first_failure.isna()]
        eligible=data.eligible_fit.to_numpy();y=data.loc[eligible,'fails_next_step'];s=scores[eligible]
        results.append(dict(method=method,representation='L/G/D + trends (frozen pilot)',n_trajectories=len(outcome),n_failures=len(fail),n_nonfailures=len(nonfail),
            n_early=int(fail.early_warning.sum()),n_false=int(nonfail.false_alarm.sum()),early_warning_rate=fail.early_warning.mean(),false_alarm_rate=nonfail.false_alarm.mean(),
            mean_lead_all_failures=fail.credited_lead_steps.mean(),auroc=roc_auc_score(y,s),aupr=average_precision_score(y,s)))
        outcomes.append(outcome.assign(method=method,representation='L/G/D + trends (frozen pilot)'))
    combined=pd.concat([pd.read_csv(ROOT/'comparison.csv'),pd.DataFrame(results)],ignore_index=True)
    combined.to_csv(ROOT/'comparison_with_lgd.csv',index=False)
    pd.concat(outcomes).to_csv(ROOT/'lgd_trajectories.csv',index=False)
    (ROOT/'lgd_comparison_provenance.json').write_text(json.dumps(dict(scope='Exploratory fixed-model comparison',
        cohort='Exactly the 168 trajectories retained by the fresh Pearson evaluator',fitting_images=281,
        no_refitting=True,no_threshold_retuning=True,frozen_models_sha256=evidence,
        thresholds={method:float(thresholds[(thresholds.method==method)&(thresholds.representation=='L/G/D + trends')].threshold.iloc[0]) for method in ['ig','gradcam']}),indent=2))
    print(combined.to_string(index=False))

if __name__=='__main__':main()
