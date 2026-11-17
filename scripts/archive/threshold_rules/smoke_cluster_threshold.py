"""Exploratory calibration-only two-cluster threshold for frozen warning scores."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from src.attribution_features import add_features
from scripts.run_expanded_cifar10c_study import score_model,save,sha
from scripts.evaluate_early_warnings import trajectory_outcomes

ROOT=Path('results/clustering_threshold_smoke')
PRIMARY=Path('results/expanded_cifar10c_5000dev_2000test')
NAMES=['IG dynamic','GRADCAM dynamic','MSP']


def cluster_threshold(scores):
    scores=np.asarray(scores,dtype=float).reshape(-1,1)
    if len(scores)<2 or not np.isfinite(scores).all() or len(np.unique(scores))<2:
        raise ValueError('Need at least two distinct finite calibration scores.')
    fitted=KMeans(n_clusters=2,n_init=20,random_state=42).fit(scores)
    centers=np.sort(fitted.cluster_centers_[:,0])
    return float(centers.mean()),centers.tolist()


def metrics(data,scores,threshold):
    outcomes=trajectory_outcomes(data,scores,threshold)
    fail=outcomes[outcomes.first_failure.notna()];nonfail=outcomes[outcomes.first_failure.isna()]
    return dict(n_failures=len(fail),n_nonfailures=len(nonfail),n_early=int(fail.early_warning.sum()),n_false=int(nonfail.false_alarm.sum()),
        early_warning_rate=float(fail.early_warning.mean()),false_alarm_rate=float(nonfail.false_alarm.mean()),
        mean_lead_all_failures=float(fail.credited_lead_steps.mean()))


def main():
    folder=PRIMARY/'frozen/resnet50';artifact=json.loads((folder/'specifications.json').read_text())
    lock=json.loads((folder/'lock.json').read_text())
    assert sha(folder/'warning_models.joblib')==lock['models_sha256'] and sha(folder/'specifications.json')==lock['specifications_sha256']
    assert sha(ROOT/'input/development.csv')==artifact['development_sha256']
    dev,_=add_features(pd.read_csv(ROOT/'input/development.csv'))
    calibration=dev[dev.image_group.isin(artifact['calibration_image_groups'])].copy()
    eligible=calibration.eligible_fit.to_numpy()
    models=joblib.load(folder/'warning_models.joblib');specs=[]
    for spec in artifact['methods']:
        if spec['name'] not in NAMES:continue
        scores=score_model(spec,models,calibration)
        threshold,centers=cluster_threshold(scores[eligible])
        specs.append(dict(name=spec['name'],original_threshold=spec['threshold'],cluster_threshold=threshold,cluster_centers=centers,
            calibration_observations=int(eligible.sum()),next_step_failures=int(calibration.loc[eligible,'fails_next_step'].sum()),
            calibration_original=metrics(calibration,scores,spec['threshold']),calibration_cluster=metrics(calibration,scores,threshold)))
    # Fit thresholds without test data or correctness labels entering clustering.
    save(ROOT/'frozen_thresholds.json',dict(scope='ResNet-only exploratory smoke test after prior blur analyses; not untouched confirmation',
        clustering='Unweighted KMeans k2 n_init20 random_state42 on eligible still-correct calibration observation scores, no resampling or labels',
        excluded='Severity0, already-failed observations, final nonfailure horizon excluded via original eligible_fit definition',
        warning_models_sha256=sha(folder/'warning_models.joblib'),primary_specifications_sha256=sha(folder/'specifications.json'),
        development_sha256=sha(ROOT/'input/development.csv'),methods=specs))
    test,_=add_features(pd.read_csv(ROOT/'input/test.csv'))
    rows=[]
    for entry in specs:
        spec=next(s for s in artifact['methods'] if s['name']==entry['name'])
        scores=score_model(spec,models,test)
        for label,key in [('Original 5%-target calibration','original_threshold'),('Two-cluster calibration midpoint','cluster_threshold')]:
            rows.append(dict(model=entry['name'],threshold_rule=label,threshold=entry[key],**metrics(test,scores,entry[key])))
    result=pd.DataFrame(rows);result.to_csv(ROOT/'comparison.csv',index=False)
    print(result.to_string(index=False))

if __name__=='__main__':main()
