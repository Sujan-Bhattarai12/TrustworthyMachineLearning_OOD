"""Exploratory sparse-map transfer check with development-frozen models/thresholds."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score,average_precision_score
from src.attribution_features import add_features
from scripts.archive.map_similarity.local_sparse_map_pilot import ROOT,distances
from scripts.evaluate_early_warnings import TRAJECTORY,COMPONENTS,trajectory_outcomes

OUT=ROOT/'gaussian_blur_transfer'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def setup():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'frozen_transfer_protocol.json'
    if path.exists():raise ValueError('Transfer protocol already frozen.')
    files=list(ROOT.glob('*.joblib'))+list(ROOT.glob('*_sparse_encoder.pt'))+[ROOT/'protocol.json',ROOT/'comparison.csv',ROOT/'calibrated_operating_points.csv',Path(__file__)]
    spec=dict(scope='Exploratory transfer to official Gaussian blur; main study blur outcomes previously examined; not untouched confirmation',
        no_refitting=True,no_test_threshold_selection=True,primary_target_calibration_fpr=.05,
        supplementary_targets=[.01,.02,.05,.1,.2],
        frozen_sha256={str(p):sha(p) for p in files})
    path.write_text(json.dumps(spec,indent=2))

def run():
    frozen=json.loads((OUT/'frozen_transfer_protocol.json').read_text())
    for path,digest in frozen['frozen_sha256'].items():
        if sha(path)!=digest:raise ValueError(f'Frozen input changed: {path}')
    data,excluded=add_features(pd.read_csv(OUT/'input/images.csv'));data=data.reset_index(drop=True)
    if set(data.corruption)!={'gaussian_blur'}:raise ValueError('Expected official Gaussian blur only.')
    original=pd.read_csv(OUT/'input/images.csv')
    if set(original.dataset)!={'cifar10c'} or set(original.ig_steps)!={64}:raise ValueError('Unexpected attribution protocol.')
    development=json.loads((ROOT/'protocol.json').read_text())
    if set(data.image_group)&set(development['fitting_images']+development['calibration_images']+development['validation_images']):
        raise ValueError('Development/test image overlap.')
    thresholds=pd.read_csv(ROOT/'calibrated_operating_points.csv');scores={}
    torch.set_num_threads(4)
    for method in ['ig','gradcam']:
        saved=torch.load(ROOT/f'{method}_sparse_encoder.pt',map_location='cpu',weights_only=True)
        encoder=nn.Sequential(nn.Linear(saved['input_dimension'],64),nn.ReLU());encoder.load_state_dict(saved['encoder']);encoder.eval()
        maps=[]
        for r in data.itertuples():
            with np.load(OUT/'input/maps'/f'{r.source}_{r.image_id}.npz',allow_pickle=False) as archive:
                a=archive[f'{r.corruption}_0_{method}'].reshape(-1)
                b=archive[f'{r.corruption}_{r.severity}_{method}'].reshape(-1)
                maps.append(np.log1p(np.maximum(b,0)/max(float(a.max()),1e-12)))
        x=np.asarray(maps,dtype=np.float32)/max(saved['rms'],1e-12)
        with torch.inference_mode():latent=encoder(torch.from_numpy(x)).numpy()
        pca=joblib.load(ROOT/f'{method}_pca.joblib');pca_latent=pca.transform(x)
        base=[f'{method}_{c}' for c in COMPONENTS]
        for name,features in [('L/G/D + trends',data[base+[f'delta_{k}' for k in base]].to_numpy()),
                              ('Sparse encoding distances',distances(latent,data)),('PCA encoding distances',distances(pca_latent,data))]:
            model=joblib.load(ROOT/f'{method}_{name.split()[0].replace("/","_")}_warning.joblib')
            scores[(method,name)]=model.predict_proba(features)[:,1]
    scores[('baseline','MSP')]=data.unconfidence.to_numpy();scores[('baseline','ODIN')]=data.odin.to_numpy()
    results=[];trajectories=[]
    for (method,name),values in scores.items():
        for row in thresholds[(thresholds.method==method)&(thresholds.representation==name)].itertuples():
            outcome=trajectory_outcomes(data,values,row.threshold)
            failing=outcome[outcome.first_failure.notna()];nonfail=outcome[outcome.first_failure.isna()]
            mask=data.eligible_fit.to_numpy();y=data.loc[mask,'fails_next_step'];s=values[mask]
            results.append(dict(method=method,representation=name,target_calibration_fpr=row.target_calibration_fpr,calibration_fpr=row.calibration_fpr,
                threshold=row.threshold,n_trajectories=len(outcome),n_failures=len(failing),n_nonfailures=len(nonfail),
                n_early=int(failing.early_warning.sum()),n_false=int(nonfail.false_alarm.sum()),early_warning_rate=failing.early_warning.mean(),
                false_alarm_rate=nonfail.false_alarm.mean(),mean_lead_all_failures=failing.credited_lead_steps.mean(),
                auroc=roc_auc_score(y,s),aupr=average_precision_score(y,s)))
            if row.target_calibration_fpr==.05:
                trajectories.append(outcome.assign(method=method,representation=name))
    result=pd.DataFrame(results);result.to_csv(OUT/'all_calibrated_targets.csv',index=False)
    primary=result[result.target_calibration_fpr==.05];primary.to_csv(OUT/'comparison.csv',index=False)
    pd.concat(trajectories).to_csv(OUT/'trajectories.csv',index=False);excluded.to_csv(OUT/'excluded.csv',index=False)
    print(primary.to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['setup','run']);args=parser.parse_args()
    (setup if args.action=='setup' else run)()
